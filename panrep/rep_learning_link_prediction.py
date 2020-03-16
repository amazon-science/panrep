"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from node_masking import  node_masker,unmask_nodes

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import pickle
from classifiers import ClassifierRGCN,ClassifierMLP
from load_data import load_kaggle_shoppers_data, load_wn_data,load_imdb_data,load_link_pred_wn_data
from model import PanRepRGCNHetero
import utils
from classifiers import DLinkPredictor as DownstreamLinkPredictor
from edge_samling import hetero_edge_masker_sampler,create_edge_mask,negative_sampling,unmask_edges
from encoders import EncoderRelGraphConvHetero,EncoderRelGraphAttentionHetero
import os
def main(args):
    rgcn_hetero(args)
def save_embeddings(model,dataset,embeddings,g,h_hid):
    rel_embeddings = torch.zeros((len(model.linkPredictor.w_relation.keys()),h_hid))
    node_embeddings=embeddings[0]
    for i, etype in enumerate(g.etypes):
        rel_embeddings[i,:]=model.linkPredictor.w_relation[etype].squeeze()
    path='../kg/ckpts/'
    emb_fname = os.path.join(path, dataset+'_PanRep' + '_relation.npy')
    np.save(emb_fname, rel_embeddings.data.cpu().numpy())
    emb_fname = os.path.join(path, dataset+'_PanRep' + '_entity.npy')
    np.save(emb_fname, node_embeddings.data.cpu().numpy())
def rgcn_hetero(args):
    if args.dataset == "wn18":
        train_edges, test_edges, valid_edges, g, masked_node_types = load_link_pred_wn_data(args)
    else:
        raise NotImplementedError


    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        # labels = labels.cuda()
        # train_idx = train_idx.cuda()
        # test_idx = test_idx.cuda()


    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
        # create model
    #dgl.contrib.sampling.sampler.EdgeSampler(g['customer_to_transaction'], batch_size=100)
    use_reconstruction_loss = args.use_reconstruction_loss
    use_infomax_loss = args.use_infomax_loss
    num_masked_nodes = args.n_masked_nodes
    node_masking = args.node_masking
    link_prediction = args.link_prediction
    mask_links = args.mask_links
    loss_over_all_nodes = args.loss_over_all_nodes
    num_sampled_edges = args.n_masked_links
    negative_rate = args.negative_rate
    #g.adjacency_matrix(transpose=True,scipy_fmt='coo',etype='customer_to_transaction')



    # for the embedding layer
    in_size_dict={}
    for name in g.ntypes:
        in_size_dict[name] = g.nodes[name].data['features'].size(1);
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i
    if args.encoder=='RGCN':
        encoder=EncoderRelGraphConvHetero(
                                  args.n_hidden,
                                  num_bases=args.n_bases,
                                  in_size_dict=in_size_dict,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=args.n_layers - 2,
                                  dropout=args.dropout,
                                  use_self_loop=args.use_self_loop)
    elif args.encoder=='RGAT':
        encoder = EncoderRelGraphAttentionHetero(
                                            args.n_hidden,
                                            in_size_dict=in_size_dict,
                                            etypes=g.etypes,
                                            ntypes=g.ntypes,
                                            num_hidden_layers=args.n_layers - 2,
                                            dropout=args.dropout,
                                            use_self_loop=args.use_self_loop)

    model = PanRepRGCNHetero(
                             args.n_hidden,
                             args.n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=args.n_layers - 2,
                             dropout=args.dropout,
                             in_size_dict=in_size_dict,
                             masked_node_types=masked_node_types,
                             loss_over_all_nodes=loss_over_all_nodes,
                             use_infomax_task=use_infomax_loss,
                             use_reconstruction_task=use_reconstruction_loss,
                             link_prediction_task=link_prediction,
                             use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    nm_time = []
    un_mas_time=[]
    lm_time = []
    backward_time = []
    model.train()



    # This creates an all 1 mask for link prediction needed by the current implementation
    ng=create_edge_mask(g,use_cuda)
    # keep a reference to previous graph?
    og=g
    g=ng
    for epoch in range(args.n_epochs):
        t_nm_0 = time.time()
        if node_masking and use_reconstruction_loss:
            masked_nodes, g = node_masker(g, num_nodes=num_masked_nodes, masked_node_types=masked_node_types)
        else:
            masked_nodes = {}
        t_nm_1 = time.time()
        # TODO check that g is not the same as g...
        t_lm_0 = time.time()
        if link_prediction:
            g, samples_d, llabels_d = hetero_edge_masker_sampler(g, num_sampled_edges, negative_rate,
                                                                     mask_links)
            link_labels_d = {}
            for e in llabels_d.keys():
                link_labels_d[e] = torch.from_numpy(llabels_d[e])
                if use_cuda:
                    link_labels_d[e] = link_labels_d[e].cuda()
            llabels_d = link_labels_d
        else:
            # TODO check that the g deletes old masked edges, nodes.
            samples_d = {}
            llabels_d = {}
        t_lm_1 = time.time()

        optimizer.zero_grad()
        t0 = time.time()
        loss, embeddings = model(g=g, masked_nodes=masked_nodes, sampled_links=samples_d,
                                 sampled_link_labels=llabels_d)
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()
        t_unmask0 = time.time()
        if node_masking and use_reconstruction_loss:
            g = unmask_nodes(g, masked_node_types)
        if link_prediction:
            g = unmask_edges(g, use_cuda)
        t_unmask1 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        lm_time.append(t_lm_1 - t_lm_0)
        nm_time.append(t_nm_1 - t_nm_0)
        un_mas_time.append(t_unmask1 - t_unmask0)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f} | Link masking Time(s) {:.4f} | "
              "Node masking Time(s) {:.4f} | Un masking time  {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1], lm_time[-1], nm_time[-1], un_mas_time[-1]))
        print("Train Loss: {:.4f}".
              format(loss.item()))

    print()

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder.forward(g)
    save_embeddings(model, args.dataset, embeddings, g, args.n_hidden)
    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
    print("Mean link masking time: {:4f}".format(np.mean(lm_time[len(lm_time) // 4:])))
    print("Mean node masking time: {:4f}".format(np.mean(nm_time[len(nm_time) // 4:])))
    ###
    # Use the encoded features for classification
    # Here we initialize the features using the reconstructed ones
    inp_dim = args.n_hidden
    nbr_downstream_layers = 1
    pan_rep_model=model
    eval_pan_rep=True
    if eval_pan_rep:
        if use_cuda:
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i].cpu()
            for e in pan_rep_model.linkPredictor.w_relation.keys():
                pan_rep_model.linkPredictor.w_relation[e] = pan_rep_model.linkPredictor.w_relation[e].cpu()

        print('original embedding')
        or_mrr = utils.calc_mrr(g, embeddings, pan_rep_model.linkPredictor.w_relation,
                                valid_edges, hits=[1, 3, 10], eval_bz=args.eval_batch_size)
        if use_cuda:
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i].cuda()
    model = DownstreamLinkPredictor(in_dim=inp_dim,
                                    out_dim=inp_dim, use_cuda=use_cuda,
                                    ntype2id=ntype2id,
                                    etypes=g.etypes,num_hidden_layers=nbr_downstream_layers)

    if use_cuda:
        model.cuda()

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    best_mrr = 0
    model_state_file = 'model_state.pth'
    for epoch in range(args.n_cepochs):
        optimizer.zero_grad()
        t_lm_0 = time.time()
        samples_d, labels_d = negative_sampling(g,train_edges,  negative_rate=args.negative_rate_downstream)
        n_labels_d = {}
        for e in labels_d.keys():
                n_labels_d[e] = torch.from_numpy(labels_d[e])
                if use_cuda:
                    n_labels_d[e] = n_labels_d[e].cuda()
        labels_d = n_labels_d
        t_lm_1 = time.time()

        t0 = time.time()
        embed = model(g,embeddings)
        loss = model.get_loss(g,embed, samples_d, labels_d)
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
                    g.nodes[ntype].data['features'] = g.nodes[ntype].data['features'].cpu()
                for etype in g.etypes:
                    g.edges[etype].data['mask'] = g.edges[etype].data['mask'].cpu()
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
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=250,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=1,
            help="gpu")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=300,
            help="number of training epochs for decoder")
    parser.add_argument("-ec", "--n-cepochs", type=int, default=500,
                        help="number of training epochs for classification")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=100,
                        help="number of masked nodes")
    parser.add_argument("-num_masked_links", "--n-masked-links", type=int, default=100,
                        help="number of masked links")
    parser.add_argument("-negative_rate", "--negative-rate", type=int, default=2,
                        help="number of negative examples per masked link")
    parser.add_argument("-negative_rate_d", "--negative-rate-downstream", type=int, default=2,
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
    parser.add_argument("--use-infomax-loss", default=False, action='store_true',
                        help="use infomax task supervision")
    parser.add_argument("--use-reconstruction-loss", default=False, action='store_true',
                        help="use feature reconstruction task supervision")
    parser.add_argument("--node-masking", default=False, action='store_true',
                        help="mask as subset of node features")
    parser.add_argument("--loss-over-all-nodes", default=False, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--link-prediction", default=True, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--mask-links", default=False, action='store_true',
                       help="mask the links to be predicted")
    parser.add_argument("--evaluate-every", type=int, default=100,
            help="perform evaluation every n epochs")
    parser.add_argument("--eval-batch-size", type=int, default=100,
            help="batch size when evaluating")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'wn18','--encoder', 'RGCN'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
