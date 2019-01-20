# Tree-Structured Long Short-Term Memory Networks

The [original implementation](https://github.com/dasguptar/treelstm.pytorch) for paper [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](http://arxiv.org/abs/1503.00075) doesn't support batch calculation of TreeLSTM.

To run the model with batch TreeLSTM:
```
- python main.py --use_batch --batchsize 25
```
which should give you exact results as without batch, but much faster in training and inference.
