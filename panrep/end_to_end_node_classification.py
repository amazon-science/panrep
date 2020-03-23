"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse
from edge_masking_samling import create_edge_mask
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl
from classifiers import ClassifierRGCN,End2EndClassifierRGCN
from load_data import load_hetero_data
from sklearn.metrics import roc_auc_score
from node_supervision_tasks import reconstruction_loss
import os

def main(args):
    rgcn_hetero(args)

def rgcn_hetero(args):
    train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_hetero_data(args)

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        labels = labels.cuda()
    #labels=labels.type(torch.LongTensor)
        #train_idx = train_idx.cuda()
        #test_idx = test_idx.cuda()

    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
        # create model
    in_size_dict={}
    g = create_edge_mask(g, use_cuda)
    for name in g.ntypes:
        in_size_dict[name] = g.nodes[name].data['features'].size(1);

    model = End2EndClassifierRGCN(
                           h_dim=args.n_hidden,
                           out_dim=num_classes,
                           num_rels = len(set(g.etypes)),
                            rel_names=list(set(g.etypes)),
                           num_bases=args.n_bases,
        ntypes=g.ntypes,in_size_dict=in_size_dict,
                           num_hidden_layers=args.n_layers - 1,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    train_indices = torch.tensor(train_idx).to(device);
    valid_indices = torch.tensor(val_idx).to(device);
    test_indices = torch.tensor(test_idx).to(device);
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(args.n_epochs):

        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g)[category_id]
        loss = F.binary_cross_entropy_with_logits(logits[train_idx].squeeze(1),
                                                  labels[train_idx].type(torch.FloatTensor).to(device))
        loss.backward()
        optimizer.step()
        # TODO is this step slowing down because of CPU?
        pred = torch.sigmoid(logits).detach().cpu().numpy()
        train_acc = roc_auc_score(labels.cpu()[train_indices.cpu()].numpy(), pred[train_indices.cpu()])
        val_acc = roc_auc_score(labels.cpu()[valid_indices.cpu()].numpy(), pred[valid_indices.cpu()])
        test_acc = roc_auc_score(labels.cpu()[test_indices.cpu()].numpy(), pred[test_indices.cpu()])

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),
            ))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.3,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=10,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=4,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=100,
            help="number of training epochs")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=1000,
                        help="number of masked nodes")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'imdb'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)