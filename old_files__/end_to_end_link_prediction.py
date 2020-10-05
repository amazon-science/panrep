"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from node_sampling_masking import  node_masker

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl
from classifiers import ClassifierRGCN,ClassifierMLP
from load_data import load_kaggle_shoppers_data, load_wn_data,load_imdb_data,load_link_pred_wn_data
from model import PanRepHetero
import utils
from classifiers import DLinkPredictor as DownstreamLinkPredictor
from edge_masking_samling import hetero_edge_masker_sampler,create_edge_mask,unmask_edges
from encoders import EncoderRelGraphConvHetero,EncoderRelGraphAttentionHetero
def main(args):
    rgcn_hetero(args)

def rgcn_hetero(args):
    if args.dataset == "wn":
        train_edges, test_edges, valid_edges, g, featless_node_types = load_link_pred_wn_data(args)
    else:
        raise NotImplementedError


    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    ng = create_edge_mask(g, use_cuda)
    # keep a reference to previous graph?
    og = g
    g = ng
    h_dim = args.n_hidden
    nbr_downstream_layers = args.n_layers-1
    # map input features as embeddings
    embeddings=[[]*len(g.ntypes)]
    for i, ntype in enumerate(g.ntypes):
        embeddings[i]=g.nodes[ntype].data['features']
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i
    model = DownstreamLinkPredictor(in_dim=embeddings[i].shape[1],out_dim=h_dim,  use_cuda=use_cuda,
                                    etypes=g.etypes, ntype2id=ntype2id,num_hidden_layers=nbr_downstream_layers,
                                    reg_param=args.regularization)

    if use_cuda:
        model.cuda()

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    pct_masked_edges=0.5

    best_mrr = 0
    model_state_file = 'end_to_end_model_state.pth'
    for epoch in range(args.n_cepochs):
        optimizer.zero_grad()
        t_lm_0 = time.time()
        # edge masking
        g, samples_d, labels_d = hetero_edge_masker_sampler(g, pct_masked_edges, args.negative_rate,
                                                             args.mask_links, use_cuda)
        t_lm_1 = time.time()

        t0 = time.time()
        embed = model(g,embeddings)
        loss = model.get_loss(g,embed, samples_d, labels_d)
        # edge unmasking
        g = unmask_edges(g, use_cuda)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
                for i in range(len(embeddings)):
                    embeddings[i]=embeddings[i].cpu()
                for ntype in g.ntypes:
                    g.nodes[ntype].data['features']=g.nodes[ntype].data['features'].cpu()
                for etype in g.etypes:
                    g.edges[etype].data['mask']=g.edges[etype].data['mask'].cpu()
                for e in model.w_relation.keys():
                    model.w_relation[e]=model.w_relation[e].cpu()
            model.eval()
            print("start eval")
            embed = model(g,embeddings)
            mrr = utils.calc_mrr(g,embed, model.w_relation, valid_edges,
                                 hits=[1, 3, 10], eval_bz=args.eval_batch_size)

            # save best model
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           model_state_file)
            if use_cuda:
                model.cuda()
                for i in range(len(embeddings)):
                    embeddings[i]=embeddings[i].cuda()
                for ntype in g.ntypes:
                    g.nodes[ntype].data['features'] = g.nodes[ntype].data['features'].cuda()
                for etype in g.etypes:
                    g.edges[etype].data['mask']=g.edges[etype].data['mask'].cuda()
                for e in model.w_relation.keys():
                    model.w_relation[e]=model.w_relation[e].cuda()

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    print("\nstart testing:")
    # use best model checkpoint
    checkpoint = torch.load(model_state_file)
    if use_cuda:
        model.cpu()  # test on CPU
        for i in range(len(embeddings)):
            embeddings[i] = embeddings[i].cpu()
        for ntype in g.ntypes:
            g.nodes[ntype].data['features'] = g.nodes[ntype].data['features'].cpu()
        for etype in g.etypes:
            g.edges[etype].data['mask'] = g.edges[etype].data['mask'].cpu()
        for e in model.w_relation.keys():
            model.w_relation[e] = model.w_relation[e].cpu()
    model.eval()
    model.load_state_dict(checkpoint['state_dict'])
    print("Using best epoch: {}".format(checkpoint['epoch']))
    embed = model(g,embeddings)
    utils.calc_mrr(g, embed, model.w_relation, test_edges,
                         hits=[1, 3, 10], eval_bz=args.eval_batch_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep')
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=200,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=2,
            help="gpu")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=4,
            help="number of propagation rounds")
    parser.add_argument("--regularization", type=float, default=0.01,
            help="regularization weight")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
            help="number of training epochs for decoder")
    parser.add_argument("-ec", "--n-cepochs", type=int, default=6000,
                        help="number of training epochs for classification")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=100,
                        help="number of masked nodes")
    parser.add_argument("-num_masked_links", "--n-masked-links", type=int, default=1000,
                        help="number of masked links")
    parser.add_argument("-negative_rate", "--negative-rate", type=int, default=10,
                        help="number of negative examples per masked link")
    parser.add_argument("-negative_rate_d", "--negative-rate-downstream", type=int, default=10,
                        help="number of negative examples per masked link for the downstream task")

    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("-en", "--encoder", type=str, required=True,
                        help="Encoder to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--use-infomax-loss", default=True, action='store_true',
                        help="use infomax task supervision")
    parser.add_argument("--use-reconstruction-loss", default=False, action='store_true',
                        help="use feature reconstruction task supervision")
    parser.add_argument("--node-masking", default=False, action='store_true',
                        help="mask as subset of node features")
    parser.add_argument("--loss-over-all-nodes", default=False, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--link-prediction", default=True, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--mask-links", default=True, action='store_true',
                       help="mask the links to be predicted")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'wn','--encoder', 'RGCN'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
