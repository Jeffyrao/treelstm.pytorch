import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import cuda

from . import Constants
from .tree import Tree


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.bsz = 128
        self.max_num_children = 10
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        # inputs: in_dim
        # child_c: num_children * mem_dim
        # child_h: num_children * mem_dim
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def batch_node_forward(self, inputs, child_c, child_h, num_children):
        # inputs: bsz * in_dim
        # child_c: bsz * max_num_child * in_dim
        # child_h: bsz * max_num_child * in_dim
        # num_children: bsz * num_children
        bsz, max_num_children, _ = child_c.size()
        child_h_sum = torch.sum(child_h, dim=1, keepdim=False)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        fh, fx = self.fh(child_h), self.fx(inputs)
        for idx in range(bsz):
            fh[idx, :num_children[idx]] += fx[idx].repeat(num_children[idx], 1)
        f = F.sigmoid(fh)

        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=1, keepdim=False)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        if isinstance(tree, list):
            return self.batch_forward(tree, inputs)
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


    def update_leaf_states(self, trees, inputs):
        queue = []
        for tree_idx, tree in enumerate(trees):
            for node_idx, node in tree.items():
                if node.num_children == 0:
                    node.visited = True
                    child_c = Variable(torch.zeros(1, self.mem_dim), requires_grad=True)
                    child_h = Variable(torch.zeros(1, self.mem_dim), requires_grad=True)
                    if cuda.is_available():
                        child_c = child_c.cuda()
                        child_h = child_h.cuda()
                    input = inputs[tree_idx][node_idx]
                    queue.append((tree_idx, node_idx, input, child_c, child_h))

        head = 0
        while head < len(queue):
            idxes, encoder_inputs, children_c, children_h = [], [], [], []
            while head < len(queue) and len(encoder_inputs) < self.bsz:
                tree_idx, node_idx, input, child_c, child_h = queue[head]
                encoder_inputs.append(input.unsqueeze(0))
                children_c.append(child_c.unsqueeze(0))
                children_h.append(child_h.unsqueeze(0))
                head += 1
            encoder_inputs = torch.cat(encoder_inputs, dim=0)
            children_c = torch.cat(children_c, dim=0)
            children_h = torch.cat(children_h, dim=0)
            num_children = [1] * len(encoder_inputs)
            #print('leaf', len(encoder_inputs))
            batch_c, batch_h = self.batch_node_forward(encoder_inputs, children_c, children_h, num_children)
            for i in range(batch_c.shape[0]):
                idx = head - batch_c.shape[0] + i
                tree_idx, node_idx, _, _, _ = queue[idx]
                trees[tree_idx][node_idx].state = (batch_c[i], batch_h[i])
        for tree_idx, tree in enumerate(trees):
            for node_idx, node in tree.items():
                if node.num_children == 0:
                    assert node.state is not None
        return queue


    def update_internal_node_states(self, trees, inputs, queue):
        head = 0
        num_internal_nodes, depth = 0, 1
        while head < len(queue):
            # find updatable parent nodes, and push to the end of queue
            prev_num_nodes = len(queue)
            while head < prev_num_nodes:
                tree_idx, node_idx, _, _, _ = queue[head]
                parent_node = trees[tree_idx][node_idx].parent
                if parent_node is not None and not parent_node.visited:
                    can_visit = True
                    children_c, children_h = [], []
                    for child_node in parent_node.children:
                        if child_node.state is None:
                            can_visit = False
                            break
                        else:
                            c, h = child_node.state
                            children_c.append(c.unsqueeze(0))
                            children_h.append(h.unsqueeze(0))
                    if can_visit:
                        parent_node.visited = True
                        children_c_var = Variable(torch.zeros(self.max_num_children, self.mem_dim),
                                                  requires_grad=True)
                        children_h_var = Variable(torch.zeros(self.max_num_children, self.mem_dim),
                                                  requires_grad=True)
                        if cuda.is_available():
                            children_c_var = children_c_var.cuda()
                            children_h_var = children_h_var.cuda()
                        children_c_var[:len(children_c)] = torch.cat(children_c, dim=0)
                        children_h_var[:len(children_h)] = torch.cat(children_h, dim=0)
                        queue.append((tree_idx, parent_node.idx, inputs[tree_idx][parent_node.idx],
                                     children_c_var, children_h_var))
                head += 1

            depth += 1
            # update parent states
            newhead = prev_num_nodes
            while newhead < len(queue):
                encoder_inputs, children_c, children_h, num_children = [], [], [], []
                while newhead < len(queue) and len(encoder_inputs) < self.bsz:
                    tree_idx, node_idx, input, child_c, child_h = queue[newhead]
                    curr_node = trees[tree_idx][node_idx]
                    encoder_inputs.append(input.unsqueeze(0))
                    children_c.append(child_c.unsqueeze(0))
                    children_h.append(child_h.unsqueeze(0))
                    num_children.append(curr_node.num_children)
                    newhead += 1
                if len(encoder_inputs) > 0:
                    encoder_inputs = torch.cat(encoder_inputs, dim=0)
                    children_c = torch.cat(children_c, dim=0)
                    children_h = torch.cat(children_h, dim=0)
                    num_internal_nodes += len(encoder_inputs)
                    #print('internal', len(encoder_inputs))
                    batch_c, batch_h = self.batch_node_forward(
                        encoder_inputs, children_c, children_h, num_children
                    )
                    for i in range(batch_c.shape[0]):
                        idx = newhead - batch_c.shape[0] + i
                        tree_idx, node_idx, _, _, _ = queue[idx]
                        trees[tree_idx][node_idx].state = (batch_c[i], batch_h[i])
            for index in range(prev_num_nodes, len(queue)):
                tree_idx, node_idx, _, _, _ = queue[idx]
                assert trees[tree_idx][node_idx].state is not None
        #print("num of internal nodes", num_internal_nodes)


    def batch_forward(self, trees, inputs):
        # trees: list[list[tree]]
        # inputs: list[torch.Tensor(seqlen, emb_size)]
        num_nodes = {}
        for tree_idx, tree in enumerate(trees):
            for node_idx, node in tree.items():
                assert node.state is None
                assert not node.visited
                depth = node.depth()
                if depth not in num_nodes:
                    num_nodes[depth] = 0
                num_nodes[depth] += 1
        # print(num_nodes)
        queue = self.update_leaf_states(trees, inputs)
        self.update_internal_node_states(trees, inputs, queue)
        root_c, root_h = [], []
        for tree in trees:
            root = Tree.get_root(tree[0])
            root_c.append(root.state[0].unsqueeze(0))
            root_h.append(root.state[1].unsqueeze(0))
        for tree_idx, tree in enumerate(trees):
            for node_idx, node in tree.items():
                assert node.state is not None
                assert node.visited
        return torch.cat(root_c, dim=0), torch.cat(root_h, dim=0)


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False
        self.childsumtreelstm = ChildSumTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        if isinstance(ltree, list):
            return self.batch_forward(ltree, linputs, rtree, rinputs)
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.childsumtreelstm(ltree, linputs)
        rstate, rhidden = self.childsumtreelstm(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output

    def batch_forward(self, ltrees, linputs, rtrees, rinputs):
        linputs_tensor, rinputs_tensor = [], []
        for i in range(len(linputs)):
            linputs_tensor.append(self.emb(linputs[i]))
            rinputs_tensor.append(self.emb(rinputs[i]))
        lstates, lhidden = self.childsumtreelstm(ltrees, linputs_tensor)
        rstates, rhidden = self.childsumtreelstm(rtrees, rinputs_tensor)
        output = self.similarity(lstates, rstates)
        return output
