from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm import SICKDataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args


def set_optimizer(model, lr, wd):
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=lr, weight_decay=wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=lr, weight_decay=wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=lr, weight_decay=wd)
    return optimizer


def get_avg_grad(named_parameters):
    layers, avg_data, avg_grads = [], [], []
    for name, param in named_parameters:
        if (param.requires_grad) and ("bias" not in name):
            layers.append(name)
            avg_data.append(param.data.abs().mean())
            if param.grad is not None:
                avg_grads.append(param.grad.abs().mean())
    return layers, avg_data, avg_grads

# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        utils.build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> SICK vocabulary size : %d ' % vocab.size())

    # load SICK dataset splits
    train_file = os.path.join(args.data, 'sick_train.pth')
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(train_dir, vocab, args.num_classes)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    dev_file = os.path.join(args.data, 'sick_dev.pth')
    if os.path.isfile(dev_file):
        dev_dataset = torch.load(dev_file)
    else:
        dev_dataset = SICKDataset(dev_dir, vocab, args.num_classes)
        torch.save(dev_dataset, dev_file)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))
    test_file = os.path.join(args.data, 'sick_test.pth')
    if os.path.isfile(test_file):
        test_dataset = torch.load(test_file)
    else:
        test_dataset = SICKDataset(test_dir, vocab, args.num_classes)
        torch.save(test_dataset, test_file)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        args.sparse,
        args.freeze_embed)
    criterion = nn.KLDivLoss(reduce=False)

    # for words common to dataset vocab and GLOVE, use GLOVE vectors
    # for other words in dataset vocab, use random normal vectors
    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        # load glove embeddings and vocab
        glove_vocab, glove_emb = utils.load_word_vectors(
            os.path.join(args.glove, 'glove.840B.300d'))
        logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())
        emb = torch.zeros(vocab.size(), glove_emb.size(1), dtype=torch.float, device=device)
        emb.normal_(0, 0.05)
        # zero out the embeddings for padding and other special words if they are absent in vocab
        for idx, item in enumerate([Constants.PAD_WORD, Constants.UNK_WORD,
                                    Constants.BOS_WORD, Constants.EOS_WORD]):
            emb[idx].zero_()
        for word in vocab.labelToIdx.keys():
            if glove_vocab.getIndex(word):
                emb[vocab.getIndex(word)] = glove_emb[glove_vocab.getIndex(word)]
        torch.save(emb, emb_file)
    # plug these into embedding matrix inside model
    model.emb.weight.data.copy_(emb)

    model.to(device), criterion.to(device)
    optimizer = set_optimizer(model, args.lr, args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)

    init_layers, init_avg_data, init_avg_grad = get_avg_grad(model.named_parameters())
    best, last_dev_loss = -float('inf'), float('inf')
    dataset = train_dataset

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        outputs_nobatch, losses_nobatch = [], []
        lstates_nobatch, rstates_nobatch = [], []
        for idx in range(args.batchsize):
            ltree, linput, rtree, rinput, label = dataset[idx]
            lroot, ltree = ltree[0], ltree[1]
            rroot, rtree = rtree[0], rtree[1]
            target = utils.map_label_to_target(label, dataset.num_classes)
            linput, rinput = linput.to(device), rinput.to(device)
            target = target.to(device)
            linputs = model.emb(linput)
            rinputs = model.emb(rinput)
            lstate, lhidden = model.childsumtreelstm(lroot, linputs)
            rstate, rhidden = model.childsumtreelstm(rroot, rinputs)
            output = model.similarity(lstate, rstate)
            #output = model(lroot, linput, rroot, rinput)
            outputs_nobatch.append(output)
            lstates_nobatch.append(lstate)
            rstates_nobatch.append(rstate)
            loss = criterion(output, target)
            losses_nobatch.append(loss)
            total_loss += loss.sum()
            loss.sum().backward()
        print(total_loss / args.batchsize)
        layers1, avg_data1, avg_grad1 = get_avg_grad(model.named_parameters())

        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        ltrees, linputs, rtrees, rinputs, labels = dataset.get_next_batch(0, args.batchsize)
        targets = []
        for i in range(len(linputs)):
            linputs[i] = linputs[i].to(device)
            rinputs[i] = rinputs[i].to(device)
            target = utils.map_label_to_target(labels[i], dataset.num_classes)
            targets.append(target.to(device))
        targets = torch.cat(targets, dim=0)
        linputs_tensor, rinputs_tensor = [], []
        for i in range(len(linputs)):
            linputs_tensor.append(model.emb(linputs[i]))
            rinputs_tensor.append(model.emb(rinputs[i]))
        lstates, lhidden = model.childsumtreelstm(ltrees, linputs_tensor)
        rstates, rhidden = model.childsumtreelstm(rtrees, rinputs_tensor)
        outputs = model.similarity(lstates, rstates)
        losses = criterion(outputs, targets)
        total_loss += losses.sum()
        losses.sum().backward()
        layers2, avg_data2, avg_grad2 = get_avg_grad(model.named_parameters())
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    main()
