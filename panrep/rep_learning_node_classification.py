"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
'''
     g = dgl.heterograph({
        ('a', 'ab', 'b'): [(0, 1), (1, 2)],
        ('b', 'ba', 'a'): [(1, 2), (2, 3)]})
    g.nodes['a'].data['x'] = torch.randn(4, 5)
    g.nodes['b'].data['y'] = torch.randn(3, 4)
    g2 = dgl.to_homo(g)
    etype_id = g.get_etype_id('ab')
    eids_of_etype = g2.filter_edges(lambda edges: edges.data[dgl.ETYPE] == etype_id)
    sg = g2.edge_subgraph({('_N', '_E', 'N'): eids_of_etype}, preserve_nodes=True)
    ng=g
    del ng.nodes['transaction'].data['features']
    del ng.nodes['history'].data['features']
    g2 = dgl.to_homo(ng)
    etype_id = g.get_etype_id('transaction_to_customer')
    eids_of_etype = g2.filter_edges(lambda edges: edges.data[dgl.ETYPE] == etype_id)
    sg = g2.edge_subgraph({('_N','_E','N'): eids_of_etype}, preserve_nodes=True)
    '''
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
import dgl
from classifiers import ClassifierRGCN,ClassifierMLP
from load_data import load_db_data, load_gen_data
from model import PanRepRGCN,PanRepRGCNHetero
from sklearn.metrics import roc_auc_score
from node_supervision_tasks import reconstruction_loss
import utils
from encoders import EncoderRelGraphConvHetero,EncoderRelGraphAttentionHetero

def main(args):
    rgcn_hetero(args)

def rgcn_hetero(args):
    if args.dataset == "database_data":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_db_data(args)
    else:
        raise NotImplementedError


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
    use_reconstruction_loss=True
    use_infomax_loss=False
    num_masked_nodes = args.n_masked_nodes
    node_masking= False
    loss_over_all_nodes=False
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
                           num_classes,
                             encoder=encoder,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                            masked_node_types=masked_node_types,
                           loss_over_all_nodes=loss_over_all_nodes,
                           use_infomax_loss=use_infomax_loss,
                           use_reconstruction_loss=use_reconstruction_loss)

    if use_cuda:
        model.cuda()

    # optimizer

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):

        if node_masking:

            masked_nodes,new_g =utils.masked_nodes(g,num_nodes=num_masked_nodes,masked_node_types=masked_node_types)
            model.updated_graph(new_g)
        else:
            masked_nodes={}
        optimizer.zero_grad()
        t0 = time.time()
        loss, embeddings = model(masked_nodes=masked_nodes)
        t1 = time.time()
        loss.backward()
        optimizer.step()
        if node_masking:
            model.updated_graph(g)
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        print("Train Loss: {:.4f}".
              format(loss.item()))

    print()

    model.eval()
    with torch.no_grad():
        embeddings = model.encoder.forward()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
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
    train_indices = torch.tensor(train_idx).to(device);
    valid_indices = torch.tensor(val_idx).to(device);
    test_indices = torch.tensor(test_idx).to(device);
    best_val_acc = 0
    best_test_acc = 0

    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        logits = model(feats)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx].squeeze(1), labels[train_idx].type(torch.FloatTensor).to(device))
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



def rgcn(args):
    # load graph data
    if args.dataset=="database_data":
        raise NotImplementedError
    else:
        num_nodes, num_rels, num_classes, train_idx,test_idx,\
        val_idx,labels, feats,edge_type,edge_norm,edge_src,edge_dst= load_gen_data(args)

    # edge type and normalization factor
    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
    labels = torch.from_numpy(labels).view(-1)

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
        feats = feats.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        labels = labels.cuda()

    # create graph
    g = DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(edge_src, edge_dst)
    inp_dim=len(g)
    # create model
    model = PanRepRGCN(len(g),
                    args.n_hidden,
                    inp_dim,
                    num_classes,
                    num_rels,
                    num_bases=args.n_bases,
                    num_hidden_layers=args.n_layers - 2,
                    dropout=args.dropout,
                    use_self_loop=args.use_self_loop,
                    use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        reconstructed_feats,embedding = model(g, feats, edge_type, edge_norm)
        loss = reconstruction_loss(reconstructed_feats[train_idx], feats[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        #train_acc = torch.sum(reconstructed_feats[train_idx].argmax(dim=1) == feats[train_idx]).item() / len(train_idx)
        val_loss =reconstruction_loss(reconstructed_feats[val_idx], feats[val_idx])
        #val_acc = torch.sum(reconstructed_feats[val_idx].argmax(dim=1) == feats[val_idx]).item() / len(val_idx)
        print("Train Loss: {:.4f} |  Validation loss: {:.4f}".
              format(loss.item(), val_loss.item()))
    print()

    model.eval()
    with torch.no_grad():
        reconstructed_feats,embedding = model.forward(g, feats, edge_type, edge_norm)
    test_loss = reconstruction_loss(reconstructed_feats[test_idx], feats[test_idx])
    print(" Test loss: {:.4f}".format(test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))
    ###
    # Use the encoded features for classification
    # Here we initialize the features using the reconstructed ones
    feats=embedding
    inp_dim=feats.shape[1]
    model = ClassifierRGCN(len(g),
                         args.n_hidden,
                         inp_dim,
                         num_classes,
                         num_rels,
                         num_bases=args.n_bases,
                         num_hidden_layers=args.n_layers - 2,
                         dropout=args.dropout,
                         use_self_loop=args.use_self_loop,
                         use_cuda=use_cuda)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        logits = model(g, feats, edge_type, edge_norm)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        t1 = time.time()
        loss.backward()
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
              format(epoch, forward_time[-1], backward_time[-1]))
        train_acc = torch.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = torch.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Train Accuracy: {:.4f} | Train Loss: {:.4f} | Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
              format(train_acc, loss.item(), val_acc, val_loss.item()))
    print()

    model.eval()
    logits = model.forward(g, feats, edge_type, edge_norm)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

    print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
    print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep')
    parser.add_argument("--dropout", type=float, default=0.3,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=40,
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
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=200,
                        help="number of masked nodes")
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
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'database_data','--encoder', 'RGAT'])
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