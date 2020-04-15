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
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.srcnodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb
def obtain_embeddings(model, seeds, blocks, device, labels, category, use_cuda):
    model.eval()
    #emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        lbl = lbl.cuda()
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
    logits = model.forward_mb(blocks)[category]
    return logits


def evaluate(model, seeds, blocks, device, labels, category, use_cuda):
    model.eval()
    #emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        lbl = lbl.cuda()
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
    logits = model.forward_mb(blocks)[category]
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
    fit(args)

def _fit(n_epochs, n_layers, n_hidden, n_bases, fanout, lr,dropout, use_self_loop, args):

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

    model = End2EndClassifierRGCN(
                           h_dim=n_hidden,
                           out_dim=num_classes,
                           num_rels = len(set(g.etypes)),
                            rel_names=list(set(g.etypes)),
                           num_bases=n_bases,g=g,device=device,
                           num_hidden_layers=n_layers - 1,
                           dropout=dropout,
                           use_self_loop=use_self_loop)

    if use_cuda:
        model.cuda()
    # train sampler
    node_embed={}
    for ntype in g.ntypes:
        if ntype not in masked_node_types:
            node_embed[ntype]=g.nodes[ntype].data['h_f']


    if len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))

    batch_size = 8 * 1024
    l2norm = 0.0001


    sampler = HeteroNeighborSampler(g, category, [fanout] * n_layers,device=device)
    loader = DataLoader(dataset=list(train_idx),
                        batch_size=batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)

    # validation sampler
    val_sampler = HeteroNeighborSampler(g, category, [fanout] * n_layers, True)
    _, val_blocks = val_sampler.sample_blocks(val_idx)

    # test sampler
    test_sampler = HeteroNeighborSampler(g, category, [fanout] * n_layers, True)
    _, test_blocks = test_sampler.sample_blocks(test_idx)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

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
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 3:
            t0 = time.time()

        for i, (seeds, blocks) in enumerate(loader):
            batch_tic = time.time()
            # need to copy the features
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            #emb = extract_embed(node_embed, blocks[0])
            lbl = labels[seeds[category]]
            logits = model.forward_mb(blocks)[category]
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

        val_loss, val_acc,val_acc_auc = evaluate(model, val_idx, val_blocks,device, labels, category,use_cuda)
        print("Epoch {:05d} | Valid Acc: {:.4f} |Valid Acc Auc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_acc_auc,val_loss.item(), np.average(dur)))
    print()
    svn=True
    if svn:

        model.eval()
        '''
        if use_cuda:
            model.cpu()
        with torch.no_grad():
            logits = model(g)
        feats = logits[category]
        # mlp_classifier(feats,use_cuda,args,num_classes,labels,train_idx,val_idx,test_idx,device)
        
        if use_cuda:
            model.cuda()
        '''
        feats=obtain_embeddings(model, test_idx, test_blocks, device, labels, category, use_cuda)
        labels_i = np.argmax(labels.cpu().numpy(), axis=1)
        feats=feats.cpu().detach().numpy()

        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std = evaluate_results_nc(
            feats, labels_i[test_idx], num_classes=num_classes)
    if args.model_path is not None:
            torch.save(model.state_dict(), args.model_path)

    test_loss, test_acc,test_acc_auc = evaluate(model, test_idx, test_blocks,device, labels, category,use_cuda)
    print("Test Acc: {:.4f}| Test Acc Auc: {:.4f}  | Test loss: {:.4f}".format(test_acc, test_acc_auc,test_loss.item()))
    print()

def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])]))
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std

def fit(args):
        n_epochs_list = [50,150,300]
        n_hidden_list = [50,200,400]
        n_layers_list = [2,3]#[2,3]
        n_bases_list = [30]
        lr_list = [1e-3,1e-4]
        dropout_list = [0.1,0.2]
        fanout_list = [None,5]
        use_link_prediction_list = [False]
        use_reconstruction_loss_list = [True]  # [True, False]
        use_infomax_loss_list = [True]  # [True, False]
        use_node_motif_list = [True]
        mask_links_list = [False]
        use_self_loop_list = [False]
        for n_epochs in n_epochs_list:
            for n_hidden in n_hidden_list:
                for n_layers in n_layers_list:
                    for n_bases in n_bases_list:
                        for fanout in fanout_list:
                            for lr in lr_list:
                                for dropout in dropout_list:
                                    for use_self_loop in use_self_loop_list:
                                                                _fit(n_epochs, n_layers, n_hidden, n_bases, fanout, lr,
                                                                     dropout, use_self_loop, args)
                                                                result = "RGCN Model, n_epochs {}; n_hidden {}; n_layers {}; n_bases {}; " \
                                                                         "fanout {}; lr {}; dropout {}" \
                                                                         "use_self_loop {} ".format(
                                                                    n_epochs,
                                                                    n_hidden,
                                                                    n_layers,
                                                                    n_bases,
                                                                    0,
                                                                    lr,
                                                                    dropout, use_self_loop)
                                                                print(result)

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.3,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=50,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=3,
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
    parser.add_argument("--use-clusteryes"
                        "a", type=int, default=0,
            help="No use of clusters")
    parser.add_argument("--use-node-motifs", default=False, action='store_true',
            help="do not use motifs")
    parser.add_argument("--split", type=int, default=5,
                        help="split type.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)


    args = parser.parse_args(['--dataset', 'imdb_preprocessed'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)