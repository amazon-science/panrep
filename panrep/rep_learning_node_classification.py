"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from node_masking import  node_masker

import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl
from classifiers import ClassifierRGCN,ClassifierMLP
from load_data import load_hetero_data
from model import PanRepRGCNHetero
from sklearn.metrics import roc_auc_score
from node_supervision_tasks import reconstruction_loss
from edge_samling import hetero_edge_masker_sampler,create_edge_mask
from encoders import EncoderRelGraphConvHetero,EncoderRelGraphAttentionHetero

def main(args):
    rgcn_hetero(args)

def rgcn_hetero(args):
    train_idx, test_idx, val_idx, labels, g, category, num_classes, masked_node_types=\
        load_hetero_data(args)


    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        labels = labels.cuda()
        #train_idx = train_idx.cuda()
        #test_idx = test_idx.cuda()


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




    if args.encoder=='RGCN':
        encoder=EncoderRelGraphConvHetero(g,
                                  args.n_hidden,
                                  num_bases=args.n_bases,
                                  num_hidden_layers=args.n_layers - 2,
                                  dropout=args.dropout,
                                  use_self_loop=args.use_self_loop)
    elif args.encoder=='RGAT':
        encoder = EncoderRelGraphAttentionHetero(g,
                                            args.n_hidden,
                                            num_hidden_layers=args.n_layers - 2,
                                            dropout=args.dropout,
                                            use_self_loop=args.use_self_loop)

    model = PanRepRGCNHetero(g,
                             args.n_hidden,
                             args.n_hidden,
                             encoder=encoder,
                             num_hidden_layers=args.n_layers - 2,
                             dropout=args.dropout,
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
    lm_time = []
    backward_time = []
    model.train()
    # This creates an all 1 mask for link prediction needed by the current implementation
    ng=create_edge_mask(g,use_cuda)
    # keep a reference to previous graph?
    og=g
    g=ng
    model.updated_graph(g)
    for epoch in range(args.n_epochs):
        t_nm_0 = time.time()
        if node_masking and use_reconstruction_loss:
            masked_nodes,new_g = node_masker(g, num_nodes=num_masked_nodes, masked_node_types=masked_node_types)
            model.updated_graph(new_g)
        else:
            new_g = g
            masked_nodes={}
        t_nm_1 = time.time()
        # TODO check that g is not the same as new_g...
        t_lm_0 = time.time()
        if link_prediction:
            new_g,samples_d,labels_d=hetero_edge_masker_sampler(new_g, num_sampled_edges, negative_rate, mask_links)
            model.updated_graph(new_g)

            n_labels_d={}
            for e in labels_d.keys():
                n_labels_d[e]=torch.from_numpy(labels_d[e])
                if use_cuda:
                    n_labels_d[e]=n_labels_d[e].cuda()
            labels_d=n_labels_d
        else:
            # TODO check that the new_g deletes old masked edges, nodes.
            samples_d = {}
            labels_d = {}
        t_lm_1 = time.time()

        optimizer.zero_grad()
        t0 = time.time()
        loss, embeddings = model(masked_nodes=masked_nodes,sampled_links=samples_d,sampled_link_labels=labels_d)
        t1 = time.time()
        loss.backward()
        optimizer.step()
        if node_masking or link_prediction:
            model.updated_graph(g)
            new_g=g
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        lm_time.append(t_lm_1-t_lm_0)
        nm_time.append(t_nm_1-t_nm_0)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f} | Link masking Time(s) {:.4f} | Node masking Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1],lm_time[-1],nm_time[-1]))
        print("Train Loss: {:.4f}".
              format(loss.item()))

    print()

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder.forward()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
    print("Mean link masking time: {:4f}".format(np.mean(lm_time[len(lm_time) // 4:])))
    print("Mean node masking time: {:4f}".format(np.mean(nm_time[len(nm_time) // 4:])))
    ###
    # Use the encoded features for classification
    # Here we initialize the features using the reconstructed ones
    feats = embeddings[category_id]
    inp_dim = feats.shape[1]
    model = ClassifierMLP(input_size=inp_dim, hidden_size=args.n_hidden,out_size=num_classes)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    # TODO find all zero indices rows and remove.
    zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

    train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
    val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
    test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))

    train_indices = torch.tensor(train_idx).to(device);
    valid_indices = torch.tensor(val_idx).to(device);
    test_indices = torch.tensor(test_idx).to(device);




    best_val_acc = 0
    best_test_acc = 0
    labels_n=labels

    for epoch in range(args.n_cepochs):
        optimizer.zero_grad()
        logits = model(feats)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx].squeeze(1), labels_n[train_idx].type(torch.FloatTensor).to(device))
        loss.backward()
        optimizer.step()
        # TODO is this step slowing down because of CPU?
        pred = torch.sigmoid(logits).detach().cpu().numpy()

        train_acc = roc_auc_score(labels_n.cpu()[train_indices.cpu()].numpy(),
                                  pred[train_indices.cpu()],average='macro')
        val_acc = roc_auc_score(labels_n.cpu()[valid_indices.cpu()].numpy(),
                                pred[valid_indices.cpu()],average='macro')
        test_acc = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()],average='macro')
        test_acc_w = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()], average='weighted')
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Weighted Test Acc %.4f' % (
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),test_acc_w.item()
            ))
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep')
    parser.add_argument("--dropout", type=float, default=0.3,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=30,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
            help="number of training epochs for decoder")
    parser.add_argument("-ec", "--n-cepochs", type=int, default=400,
                        help="number of training epochs for classification")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=20,
                        help="number of masked nodes")
    parser.add_argument("-num_masked_links", "--n-masked-links", type=int, default=50,
                        help="number of masked links")
    parser.add_argument("-negative_rate", "--negative-rate", type=int, default=10,
                        help="number of negative examples per masked link")


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
    parser.add_argument("--node-masking", default=True, action='store_true',
                        help="mask as subset of node features")
    parser.add_argument("--loss-over-all-nodes", default=True, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--link-prediction", default=False, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--mask-links", default=True, action='store_true',
                       help="mask the links to be predicted")

    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'imdb','--encoder', 'RGCN'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)

    '''
    # perform edge neighborhood sampling to generate training graph and data
    g, node_id, edge_type, node_norm, data, labels = \
        utils.generate_sampled_graph_and_labels(
            train_data, args.graph_batch_size, args.graph_split_size,
            num_rels, adj_list, degrees, args.negative_sample,
            args.edge_sampler)
    print("Done edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
        data, labels = data.cuda(), labels.cuda()
    '''