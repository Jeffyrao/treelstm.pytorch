import torch
from .tree import Tree
from .model import ChildSumTreeLSTM

t1_n1 = Tree()
t1_n1.idx = 0
t1_n2 = Tree()
t1_n2.idx = 1
t1_n1.add_child(t1_n2)
t1_n2.parent = t1_n1
tree1 = {0: t1_n1, 1: t1_n2}

t2_n1 = Tree()
t2_n1.idx = 0
t2_n2 = Tree()
t2_n2.idx = 1
t2_n3 = Tree()
t2_n3.idx = 2
t2_n4 = Tree()
t2_n4.idx = 3
t2_n3.add_child(t2_n1)
t2_n3.add_child(t2_n2)
t2_n1.parent = t2_n3
t2_n2.parent = t2_n3
t2_n2.add_child(t2_n4)
t2_n4.parent=t2_n2
tree2 = {0: t2_n1, 1: t2_n2, 2: t2_n3, 3: t2_n4}
trees = [tree1, tree2]

tensor1 = torch.Tensor(2, 10)
tensor2 = torch.Tensor(4, 10)
tensors = [tensor1, tensor2]

tree_lstm = ChildSumTreeLSTM(10, 4)

state1, hidden1 = tree_lstm(t1_n1, tensor1)
state2, hidden2 = tree_lstm(t2_n3, tensor2)

print("state1", state1)
print("hidden1", hidden1)
print("state2", state2)
print("hidden2", hidden2)

for tree in trees:
    for idx, node in tree.items():
        node.state = None
batch_state, batch_hidden = tree_lstm(trees, tensors)
print(batch_state)
print(batch_hidden)