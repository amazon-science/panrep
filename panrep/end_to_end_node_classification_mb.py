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
import dgl
from torch.utils.data import DataLoader

from node_sampling_masking import HeteroNeighborSampler
from classifiers import ClassifierRGCN,End2EndClassifierRGCN
from load_data import load_hetero_data
from sklearn.metrics import roc_auc_score,accuracy_score
from node_supervision_tasks import reconstruction_loss
import numpy as np

def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.srcnodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def evaluate(model, seeds, blocks, node_embed, labels, category, use_cuda):
    model.eval()
    emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        emb = {k : e.cuda() for k, e in emb.items()}
        lbl = lbl.cuda()
    logits = model.forward_mb(emb, blocks)[category]
    loss = F.binary_cross_entropy_with_logits(logits, lbl)

    acc = torch.sum(logits.argmax(dim=1) == lbl.argmax(dim=1) ).item() / len(seeds)
    pred = torch.sigmoid(logits).detach().cpu().numpy()
    try:
        acc_auc = roc_auc_score(lbl.cpu().numpy(), pred)
    except ValueError:
        acc_auc = -1
        pass
    return loss, acc,acc_auc

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
    # train sampler
    node_embed={}
    for ntype in g.ntypes:
       node_embed[ntype]=g.nodes[ntype].data['features']


    if len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))




    sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers)
    loader = DataLoader(dataset=list(train_idx),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)

    # validation sampler
    val_sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers, True)
    _, val_blocks = val_sampler.sample_blocks(val_idx)

    # test sampler
    test_sampler = HeteroNeighborSampler(g, category, [args.fanout] * args.n_layers, True)
    _, test_blocks = test_sampler.sample_blocks(test_idx)

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
    # training loop
    print("start training...")
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 3:
            t0 = time.time()

        for i, (seeds, blocks) in enumerate(loader):
            batch_tic = time.time()
            # need to copy the features

            emb = extract_embed(node_embed, blocks[0])
            lbl = labels[seeds[category]]
            if use_cuda:
                emb = {k: e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            logits = model.forward_mb(emb,blocks)[category]
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
            loss.backward()
            optimizer.step()

            train_acc = torch.sum(logits.argmax(dim=1) == lbl.argmax(dim=1)).item() / len(seeds[category])
            pred = torch.sigmoid(logits).detach().cpu().numpy()
            try:
                train_acc_auc = roc_auc_score(lbl.cpu().numpy(), pred)
            except ValueError:
                train_acc_auc=-1
                pass
            print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} |Train Acc AUC: {:.4f}| Train Loss: {:.4f} | Time: {:.4f}".
                  format(epoch, i, train_acc,train_acc_auc, loss.item(), time.time() - batch_tic))

        if epoch > 3:
            dur.append(time.time() - t0)

        val_loss, val_acc,val_acc_auc = evaluate(model, val_idx, val_blocks,node_embed, labels, category,use_cuda)
        print("Epoch {:05d} | Valid Acc: {:.4f} |Valid Acc Auc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_acc_auc,val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        torch.save(model.state_dict(), args.model_path)

    test_loss, test_acc,test_acc_auc = evaluate(model, test_idx, test_blocks,node_embed, labels, category,use_cuda)
    print("Test Acc: {:.4f}| Test Acc Auc: {:.4f}  | Test loss: {:.4f}".format(test_acc, test_acc_auc,test_loss.item()))
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
    parser.add_argument("--batch-size", type=int, default=400,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--fanout", type=int, default=20,
            help="Fan-out of neighbor sampling.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)


    args = parser.parse_args(['--dataset', 'imdb'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)