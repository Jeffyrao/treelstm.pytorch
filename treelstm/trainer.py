from tqdm import tqdm

import torch

from . import utils


def get_avg_grad(named_parameters):
    layers, avg_data, avg_grads = [], [], []
    for name, param in named_parameters:
        if (param.requires_grad) and ("bias" not in name):
            layers.append(name)
            avg_data.append(param.data.abs().mean())
            if param.grad is not None:
                avg_grads.append(param.grad.abs().mean())
    return layers, avg_data, avg_grads


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    def clear_states(self, trees):
        for tree_idx, tree in enumerate(trees):
            for node_idx, node in tree.items():
                assert node.state is not None
                assert node.visited
                node.state = None
                node.visited = False

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        if not self.args.use_batch:
            for idx in range(len(dataset)):
                ltree, linput, rtree, rinput, label = dataset[indices[idx]]
                lroot, ltree = ltree[0], ltree[1]
                rroot, rtree = rtree[0], rtree[1]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                output = self.model(lroot, linput, rroot, rinput)
                loss = self.criterion(output, target)
                total_loss += loss.sum()
                loss.sum().backward()
                if (idx + 1) % self.args.batchsize == 0 and idx > 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
        else:
            for idx in range(0, len(dataset), self.args.batchsize):
                ltrees, rtrees, linputs, rinputs, labels = [], [], [], [], []
                for new_index in indices[idx: min(len(dataset), idx+self.args.batchsize)]:
                    ltree, lsent, rtree, rsent, label = dataset[new_index]
                    ltrees.append(ltree[1])
                    rtrees.append(rtree[1])
                    linputs.append(lsent)
                    rinputs.append(rsent)
                    labels.append(label)

                targets = []
                for i in range(len(linputs)):
                    linputs[i] = linputs[i].to(self.device)
                    rinputs[i] = rinputs[i].to(self.device)
                    target = utils.map_label_to_target(labels[i], dataset.num_classes)
                    targets.append(target.to(self.device))
                targets = torch.cat(targets, dim=0)
                outputs = self.model(ltrees, linputs, rtrees, rinputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.sum()
                loss.sum().backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.clear_states(ltrees)
                self.clear_states(rtrees)
        self.epoch += 1
        return total_loss / len(dataset)

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            if not self.args.use_batch:
                indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
                for idx in range(len(dataset)):
                    ltree, linput, rtree, rinput, label = dataset[idx]
                    lroot, ltree = ltree[0], ltree[1]
                    rroot, rtree = rtree[0], rtree[1]
                    target = utils.map_label_to_target(label, dataset.num_classes)
                    linput, rinput = linput.to(self.device), rinput.to(self.device)
                    target = target.to(self.device)
                    output = self.model(lroot, linput, rroot, rinput)
                    loss = self.criterion(output, target)
                    total_loss += loss.sum()
                    output = output.squeeze().to('cpu')
                    predictions[idx] = torch.dot(indices, torch.exp(output))
            else:
                indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
                for idx in range(0, len(dataset), self.args.batchsize):
                    ltrees, linputs, rtrees, rinputs, labels = dataset.get_next_batch(idx, self.args.batchsize)
                    targets = []
                    for i in range(len(linputs)):
                        linputs[i] = linputs[i].to(self.device)
                        rinputs[i] = rinputs[i].to(self.device)
                        target = utils.map_label_to_target(labels[i], dataset.num_classes)
                        targets.append(target.to(self.device))
                    targets = torch.cat(targets, dim=0)
                    outputs = self.model(ltrees, linputs, rtrees, rinputs)
                    losses = self.criterion(outputs, targets)
                    total_loss += losses.sum()
                    outputs = outputs.to('cpu')
                    batch_indices = indices.repeat(len(ltrees), 1)
                    predictions[idx: idx+len(ltrees)] = \
                        (batch_indices * torch.exp(outputs)).sum(dim=1, keepdim=False)
                    self.clear_states(ltrees)
                    self.clear_states(rtrees)
        return total_loss / len(dataset), predictions
