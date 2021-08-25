'''
This file contains functions that help loading the different datasets in the required format.
'''

import os
import pickle
import random
import dgl.function as fn
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import dgl
import scipy.io
import urllib.request
import numpy as np
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import torch
#from aux_files.DistDGL.DistDGL.python.dgl.data import OAGDataset
from dgl.contrib.data import load_data
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from statistics import median
from scipy.cluster.vq import vq, kmeans2, whiten
import pandas as pd
import pandas as pd
from ogb.nodeproppred import DglNodePropPredDataset



def compute_cluster_assignemnts(features,cluster_number):
    centroid, label = kmeans2(features,cluster_number,minit='points')
    one_hot=pd.get_dummies(label)

    return torch.tensor(one_hot.values).float()
def generate_rwalks(g,metapaths,samples_per_node=20,device=None,rw_supervision=True):
    rw_neighbors={}
    if not rw_supervision:
        return None
    for ntype in metapaths.keys():
        if ntype in g.ntypes:
            traces,types=dgl.sampling.random_walk(g, list(np.arange(g.number_of_nodes(ntype)))* samples_per_node, metapath = metapaths[ntype])
            # remove the same node id as the start of the walk!!
            traces=traces[:,1:]
            types=types[1:]
            sampled_ntypes=list(types.numpy())*samples_per_node
            rw_neighbors_ids=traces.reshape((g.number_of_nodes(ntype),samples_per_node*traces.shape[1]))
            rw_neighbors[ntype]=(rw_neighbors_ids,sampled_ntypes)
            neighbors = rw_neighbors[ntype][0]
            neighbor_per_ntype = {}
            for id in range(len(rw_neighbors[ntype][1])):
                neighbor_type = g.ntypes[rw_neighbors[ntype][1][id]]
                if neighbor_type in neighbor_per_ntype:
                        neighbor_per_ntype[neighbor_type] = torch.cat(
                            (neighbor_per_ntype[neighbor_type], neighbors[:, id].unsqueeze(0).transpose(1, 0).to(device)), dim=1)
                else:
                        neighbor_per_ntype[neighbor_type] = neighbors[:, id].unsqueeze(0).transpose(1, 0).to(device)
            rw_neighbors[ntype]=neighbor_per_ntype

    return rw_neighbors
def load_hetero_data(args):
    if args.dataset == "kaggle_shoppers":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_kaggle_shoppers_data(args)
    elif args.dataset == "wn18":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_wn_data(args)
    elif args.dataset == "imdb":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_imdb_data(args)
    elif args.dataset == "imdb_preprocessed":
        train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types,rw_neighbors= load_imdb_preprocessed_data(args)
    elif args.dataset == "dblp_preprocessed":
        train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types,rw_neighbors= load_dblp_preprocessed_data(
            args)
    elif args.dataset == "imdb_pre_xiang":
        train_idx, test_idx, val_idx, labels, g, category, num_classes, masked_node_types = load_imdb_prexiang_preprocessed_data(
            args)
    else:
        raise NotImplementedError
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types,rw_neighbors
def load_univ_hetero_data(args):
    multilabel=False
    if args.dataset == "imdb_preprocessed":
        train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g= load_imdb_univ_preprocessed_data(args)
    elif args.dataset == "acm":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_acm_univ_data(args)
    elif args.dataset == 'aifb' or args.dataset == 'mutag' or args.dataset == 'bgs' or args.dataset == 'am':
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_std_het_full_univ_data(args)
    elif args.dataset == "oag_full":

        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_oag_full_univ_data(args)
    elif args.dataset == 'ogbn-mag':
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_ogbn_mag_full_univ_data(args)
    elif args.dataset == "oag":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_oag_univ_preprocessed_data(args)
    elif args.dataset == "oag_na":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g = load_oag_na_univ_preprocessed_data(args)
    elif args.dataset == "dblp_preprocessed":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g =load_dblp_univ_preprocessed_data(args)
    elif args.dataset == "query_biodata":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g=load_query_biodata_univ_data(args)
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g=load_query_biodata_univ_data(args)
    elif args.dataset == "drkg":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g=load_drkg_edge_few_shot_data(args)
    else:
        raise NotImplementedError
    if not args.use_node_features:
        for ntype in train_g.srctypes:
            if train_g.srcnodes[ntype].data.get('h_f', None) is not None:
                del  train_g.srcnodes[ntype].data['h_f']
            if test_g.srcnodes[ntype].data.get('h_f', None) is not None:
                del test_g.srcnodes[ntype].data['h_f']
            if valid_g.srcnodes[ntype].data.get('h_f', None) is not None:
                del valid_g.srcnodes[ntype].data['h_f']
        for ntype in train_g.dsttypes:
            if train_g.dstnodes[ntype].data.get('h_f', None) is not None:
                del  train_g.dstnodes[ntype].data['h_f']
            if test_g.srcnodes[ntype].data.get('h_f', None) is not None:
                del test_g.dstnodes[ntype].data['h_f']
            if valid_g.srcnodes[ntype].data.get('h_f', None) is not None:
                del valid_g.dstnodes[ntype].data['h_f']
    if labels is not None and len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))
    return train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g,multilabel

def hetero_data_to_homo_data(train_idx, test_idx, val_idx, labels, category, num_classes,
                             featless_node_types, rw_neighbors,
                             train_edges, test_edges, valid_edges, train_gh, valid_g, test_g):
    category_id = len(train_gh.ntypes)
    for i, ntype in enumerate(train_gh.ntypes):
        if ntype == category:
            category_id = i
    train_g = dgl.to_homogeneous(train_gh)
    node_ids = torch.arange(train_g.number_of_nodes())
    node_tids = train_g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]

    return train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g
def load_few_edge_shot_hetero_data(args):
    if args.dataset == "imdb_preprocessed":
        train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g= load_imdb_univ_preprocessed_data(args)
    elif args.dataset == "dblp_preprocessed":
        train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g= load_dblp_univ_preprocessed_data(args)
    elif args.dataset=='drkg':
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g =  load_drkg_edge_few_shot_data(args)
    else:
        raise NotImplementedError
    return train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g
def load_oag_nc_lp(args):
    dir='../data/oaggpt/oag_NN.dgl'
    dataset = dgl.load_graphs(dir)[0]
    hg = dataset[0]

    # Construct author embeddings by averaging over their papers' embeddings.
    hg.multi_update_all(
        {'rev_AP_write_first': (fn.copy_src('emb', 'm'), fn.sum('m', 'h')),
         'rev_AP_write_last': (fn.copy_src('emb', 'm'), fn.sum('m', 'h')),
         'rev_AP_write_other': (fn.copy_src('emb', 'm'), fn.sum('m', 'h')),},
        'sum')
    cnts = hg.in_degrees(etype='rev_AP_write_first') + hg.in_degrees(etype='rev_AP_write_last') + hg.in_degrees(etype='rev_AP_write_other')
    cnts = cnts.reshape(-1, 1)
    hg.nodes['author'].data['emb'] = hg.nodes['author'].data['h'] / cnts

    # Construct labels of paper nodes
    ss, dd = hg.edges(etype=('field', 'rev_PF_in_L2', 'paper'))
    ssu_, ssu = torch.unique(ss, return_inverse=True)
    print('Full label set size:', len(ssu_))
    paper_labels = torch.zeros(hg.num_nodes('paper'), len(ssu_), dtype=torch.bool)
    paper_labels[dd, ssu] = True

    # Split the dataset into training, validation and testing.
    label_sum = paper_labels.sum(1)
    times=hg.nodes['paper'].data['time']

    pre_range = {t: True for t in times.numpy() if t != None and t < 2014}
    train_range = {t: True for t in times.numpy() if t != None and t >= 2014 and t <= 2016}
    valid_range = {t: True for t in times.numpy() if t != None and t > 2016 and t <= 2017}
    test_range = {t: True for t in times.numpy() if t != None and t > 2017}

    pre_target_nodes = []
    train_target_nodes = []
    valid_target_nodes = []
    test_target_nodes = []
    target_type = 'paper'
    rel_stop_list = ['self', 'rev_PF_in_L0', 'rev_PF_in_L5', 'rev_PV_Repository', 'rev_PV_Patent']

    for p_id, _time in enumerate(times):
        if float(_time.numpy()) in pre_range:
            pre_target_nodes += [[p_id, _time]]
        elif float(_time.numpy()) in train_range:
            train_target_nodes += [[p_id, _time]]
        elif float(_time.numpy()) in valid_range:
            valid_target_nodes += [[p_id, _time]]
        elif float(_time.numpy()) in test_range:
            test_target_nodes += [[p_id, _time]]
    pre_target_nodes = np.array(pre_target_nodes)
    train_target_nodes = np.array(train_target_nodes)
    valid_target_nodes = np.array(valid_target_nodes)
    test_target_nodes = np.array(test_target_nodes)

    train_idx = torch.tensor(train_target_nodes[:, 0], dtype=int)
    val_idx = torch.tensor(valid_target_nodes[:, 0], dtype=int)
    test_idx = torch.tensor(test_target_nodes[:, 0], dtype=int)

    # Remove infrequent labels. Otherwise, some of the labels will not have instances
    # in the training, validation or test set.
    num_filter=-1
    label_filter = paper_labels[train_idx].sum(0) > num_filter
    label_filter = torch.logical_and(label_filter, paper_labels[val_idx].sum(0) > num_filter)
    label_filter = torch.logical_and(label_filter, paper_labels[test_idx].sum(0) > num_filter)
    paper_labels = paper_labels[:,label_filter]
    paper_labels=paper_labels.float()
    print('#labels:', paper_labels.shape[1])
    if args.klloss:
        paper_labels /= paper_labels.sum(axis=1).reshape(-1, 1)


    # Adjust training, validation and testing set to make sure all paper nodes
    # in these sets have labels.
    train_idx = train_idx[paper_labels[train_idx].sum(1) > 0]
    val_idx = val_idx[paper_labels[val_idx].sum(1) > 0]
    test_idx = test_idx[paper_labels[test_idx].sum(1) > 0]
    # All labels have instances.
    if num_filter>=0:
        assert np.all(paper_labels[train_idx].sum(0).numpy() > 0)
        assert np.all(paper_labels[val_idx].sum(0).numpy() > 0)
        assert np.all(paper_labels[test_idx].sum(0).numpy() > 0)
        # All instances have labels.
        assert np.all(paper_labels[train_idx].sum(1).numpy() > 0)
        assert np.all(paper_labels[val_idx].sum(1).numpy() > 0)
        assert np.all(paper_labels[test_idx].sum(1).numpy() > 0)

    # Remove field nodes from the graph.
    etypes = []
    for etype in hg.canonical_etypes:
        if etype[0] != 'field' and etype[2] != 'field':
            etypes.append(etype)
    hg = dgl.edge_type_subgraph(hg, etypes)
    print(hg.canonical_etypes)

    # Construct node features.
    # TODO(zhengda) we need to construct the node features for author nodes.
    ntypes = []
    if args.use_node_features:
        node_feats = []
        for ntype in hg.ntypes:
            print(ntype)
            if ntype != 'field' and 'emb' in hg.nodes[ntype].data:
                feat = hg.nodes[ntype].data.pop('emb')
                node_feats.append(feat.share_memory_())
                ntypes.append(ntype)
            else:
                node_feats.append(None)
    else:
        node_feats = [None] * len(hg.ntypes)
    print('nodes with features:', ntypes)
    #print(node_feats)

    category = 'paper'
    return hg, node_feats, paper_labels, train_idx, val_idx, test_idx, category, paper_labels.shape[1]


def load_univ_homo_data(args):
    ogb_dataset = False
    oag_data = False
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    elif args.dataset == 'oag_cs':
        dataset = load_oag_nc_lp(args)
        oag_data = True
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset(name=args.dataset)
        ogb_dataset = True
    else:
        raise ValueError()

    if ogb_dataset is True:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-' + etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        if args.dataset == 'ogbn-mag':
            category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

        if args.use_node_features:

            node_feats = []
            for ntype in hg.ntypes:
                if len(hg.nodes[ntype].data) == 0:
                    node_feats.append(None)
                else:
                    assert len(hg.nodes[ntype].data) == 1
                    feat = hg.nodes[ntype].data.pop('feat')
                    node_feats.append(feat.share_memory_())
        else:
            node_feats = [None] * num_of_ntype
    elif oag_data:
        hg, node_feats, labels, train_idx, val_idx, test_idx, category, num_classes = dataset

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)




    else:
        # Load from hetero-graph
        hg = dataset[0]

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        category = dataset.predict_category
        num_classes = dataset.num_classes
        train_mask = hg.nodes[category].data.pop('train_mask')
        test_mask = hg.nodes[category].data.pop('test_mask')
        labels = hg.nodes[category].data.pop('labels')
        train_idx = torch.nonzero(train_mask).squeeze()
        test_idx = torch.nonzero(test_mask).squeeze()
        node_feats = [None] * num_of_ntype

        # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
        # Split train set into train and validation if args.validation is set
        # otherwise use train set as the validation set.
        if args.validation:
            val_idx = train_idx[:len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5:]
        else:
            val_idx = train_idx

    # calculate norm for each edge type and store in edge
    if args.global_norm is False:
        for canonical_etype in hg.canonical_etypes:
            u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
            _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
            degrees = count[inverse_index]
            norm = torch.ones(eid.shape[0]) / degrees
            norm = norm.unsqueeze(1)
            hg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    g = dgl.to_homogeneous(hg, edata=['norm'])
    if args.global_norm:
        u, v, eid = g.all_edges(form='all')
        _, inverse_index, count = torch.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = torch.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        g.edata['norm'] = norm

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()
    node_ids = torch.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    cluster_assignments=[]
    if args.use_clusterandrecover_loss:
        for feat in node_feats:
            if feat is not None:
                cluster_assignments.append(compute_cluster_assignemnts(feat, cluster_number=args.num_cluster))
            else:
                cluster_assignments.append(None)

    #target_idx.share_memory_()
    #train_idx.share_memory_()
    #val_idx.share_memory_()
    #test_idx.share_memory_()
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    g.create_formats_()
    metapaths={}
    train_edges=[]
    test_edges=[]
    valid_edges=[]
    test_edges=[]
    train_g=g
    valid_g=g
    test_g=g
    multilabel=False
    if oag_data:
        multilabel=True
    return train_idx,val_idx,test_idx,target_idx,labels,num_classes,node_feats,cluster_assignments,\
           metapaths, train_edges, test_edges, valid_edges, train_g, valid_g, test_g,multilabel,num_rels

def load_kge_hetero_data(args):
    if args.dataset == "imdb_preprocessed":
            load_imdb_kge_preprocessed_data(args)
    elif args.dataset == "dblp_preprocessed":
            load_dblp_kge_preprocessed_data(args)
    elif args.dataset == "oag":
            load_oag_kge_preprocessed_data(args)
    elif args.dataset == "oag_na":
            load_oag_na_kge_preprocessed_data(args)

    else:
        raise NotImplementedError
    return
def load_hetero_link_pred_data(args):
    if args.dataset == "wn18":
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g, featless_node_types = load_link_pred_wn_pick_data(
            args)
    elif args.dataset == "query_biodata":
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g, featless_node_types = load_link_pred_query_biodata_data(
            args)
    else:
        raise NotImplementedError

    return train_edges, test_edges, valid_edges, train_g, valid_g, test_g, featless_node_types
def load_link_pred_query_biodata_data(args):
    def triplets_to_dict(edges,etype_to_canonical):
        d_e={}
        s,e,d=edges
        for sou,edg,dest in zip(s,e,d):
            edg =str(edg)
            edg=etype_to_canonical[edg]
            if edg not in d_e:
                d_e[edg]=[(sou,dest)]
            else:
                d_e[edg]+=[(sou,dest)]
        return d_e
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/query_biodata/"

    # In[13]:

    g = pickle.load(open(os.patorch.join(data_folder, 'graph.pickle'), "rb")).to(torch.device("cpu"))
    #get eid from heterograph and use dgl.edge_subgraph
    train_pct = 0.8
    val_pct= 0.1
    #train_g,valid_g,test_g,train_edges,valid_edges,test_edges=create_edge_graph_splits(g,train_pct,val_pct,data_folder)

    splits_dir=pickle.load(open(os.patorch.join(data_folder, 'splits_dir.pickle'), "rb"))
    #TODO fix this is wrong had to add all edges in the testign graph
    #
    train_g=1#splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    featless_node_types=[]

    return train_edges, test_edges, valid_edges, train_g,valid_g,test_g, featless_node_types

def create_edge_few_shot_splits(g,directory,etype,K=10,val_pct=0.01):
    if os.patorch.exists(os.patorch.join(directory, "few_shot_splits_dir"+str(K)+".pickle")):
        splits_dir = pickle.load(open(os.patorch.join(directory, "few_shot_splits_dir"+str(K)+".pickle"), "rb"))

        train_g = splits_dir['train_g']
        valid_g = splits_dir['valid_g']
        test_g = splits_dir['test_g']
        train_edges = splits_dir['train_edges']
        valid_edges = splits_dir['valid_edges']
        test_edges = splits_dir['test_edges']
    else:
        num_nodes_per_types = {}
        for ntype in g.ntypes:
            num_nodes_per_types[ntype] = g.number_of_nodes(ntype)


        train_edges = {}
        valid_edges = {}
        test_edges = {}
        valid_edgesfgraph = {}
        test_edgesfgraph = {}
        for c_etype in g.canonical_etypes:
            etyp_eids = g.all_edges(form='uv', etype=c_etype)
            n_edges = etyp_eids[0].size(0)
            perm = torch.randperm(n_edges)
            if c_etype[1] not in etype:
                train_id = perm#[:int(n_edges * train_pct)]
                val_id = []#perm[int(n_edges * train_pct):int(n_edges * (train_pct + val_pct))]
                val_id_fgraph = perm#[:int(n_edges * (train_pct + val_pct))]
                test_id = []#perm[int(n_edges * (train_pct + val_pct)):]
                test_id_fgraph = perm
            else:
                train_id = perm[:K]
                val_id = perm[K:K + int(val_pct * len(etyp_eids[0]))]
                val_id_fgraph = perm[:K + int(val_pct * len(etyp_eids[0]))]
                test_id = perm[K + int(val_pct * len(etyp_eids[0])):]
                test_id_fgraph = perm

            edges = list(tuple(zip(etyp_eids[0].cpu().numpy(), etyp_eids[1].cpu().numpy())))
            train_edges[c_etype] = [edges[i] for i in train_id.numpy().astype(int)]
            if len(val_id)>0:
                valid_edges[c_etype] = [edges[i] for i in val_id.numpy().astype(int)]
            valid_edgesfgraph[c_etype] = [edges[i] for i in val_id_fgraph.numpy().astype(int)]
            if len(test_id) > 0:
                test_edges[c_etype] = [edges[i] for i in test_id.numpy().astype(int)]
            test_edgesfgraph[c_etype] = [edges[i] for i in test_id_fgraph.numpy().astype(int)]
        train_g = dgl.heterograph(train_edges, num_nodes_per_types)
        valid_g = dgl.heterograph(valid_edgesfgraph, num_nodes_per_types)
        test_g = dgl.heterograph(test_edgesfgraph, num_nodes_per_types)

        for e in train_edges.keys():
            train_edges[e] = torch.tensor(train_edges[e]).long().transpose(1, 0)
        for e in valid_edges.keys():
            valid_edges[e] = torch.tensor(valid_edges[e]).long().transpose(1, 0)
        for e in test_edges.keys():
            test_edges[e] = torch.tensor(test_edges[e]).long().transpose(1, 0)
        for ntype in g.ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                train_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
                valid_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
                test_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
        splits_dir={"train_g":train_g,"valid_g":valid_g,"test_g":test_g,"train_edges":train_edges,
                    "valid_edges":valid_edges,"test_edges":test_edges,}
        pickle.dump(splits_dir, open(os.patorch.join(directory, "few_shot_splits_dir"+str(K)+".pickle"), "wb"),
                    protocol=4);

    return train_g,valid_g,test_g,train_edges,valid_edges,test_edges

def create_edge_graph_splits_kge(g,train_pct,val_pct,directory):

    if not os.patorch.exists(directory + "splits_dir_tr" + str(train_pct) + "_val_" + str(val_pct) + ".pickle"):

        num_nodes_per_types = {}
        for ntype in g.ntypes:
            num_nodes_per_types[ntype] = g.number_of_nodes(ntype)

        train_edges = {}
        valid_edges = {}
        test_edges = {}
        valid_edgesfgraph = {}
        test_edgesfgraph = {}
        for c_etype in g.canonical_etypes:
            etyp_eids = g.all_edges(form='uv', etype=c_etype)
            n_edges = etyp_eids[0].size(0)
            perm = torch.randperm(n_edges)
            train_id = perm[:int(n_edges * train_pct)]
            val_id = perm[int(n_edges * train_pct):int(n_edges * (train_pct + val_pct))]
            val_id_fgraph = perm[:int(n_edges * (train_pct + val_pct))]
            test_id = perm[int(n_edges * (train_pct + val_pct)):]
            test_id_fgraph = perm
            edges = list(tuple(zip(etyp_eids[0].cpu().numpy(), etyp_eids[1].cpu().numpy())))
            train_edges[c_etype] = [edges[i] for i in train_id.numpy().astype(int)]
            valid_edges[c_etype] = [edges[i] for i in val_id.numpy().astype(int)]
            test_edges[c_etype] = [edges[i] for i in test_id.numpy().astype(int)]

        def totriple(edges):
            triples = []
            for e in edges.keys():
                triples += [(uv[0], e, uv[1]) for uv in edges[e]]
            return triples
        train_triples=totriple(train_edges)
        valid_triples = totriple(valid_edges)
        test_triples = totriple(test_edges)
        with open(directory+'train'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in train_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'valid'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in valid_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'test'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in test_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
    else:
        splits_dir = pickle.load(open(os.patorch.join(directory,
                    "splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"), "rb"))

        train_g = splits_dir['train_g']
        valid_g = splits_dir['valid_g']
        test_g = splits_dir['test_g']
        train_edges = splits_dir['train_edges']
        valid_edges = splits_dir['valid_edges']
        test_edges = splits_dir['test_edges']

        def totriple(edges):
            triples = []
            for e in edges.keys():
                triples += [(uv[0], e[1], uv[1]) for uv in list(map(list, zip(*edges[e].tolist())))]
            return triples
        train_triples=totriple(train_edges)
        valid_triples = totriple(valid_edges)
        test_triples = totriple(test_edges)
        with open(directory+'train'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in train_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'valid'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in valid_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'test'+str(round(0.975- train_pct,2))+'.txt', 'w') as f:
            for item in test_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))

    return
def create_edge_graph_few_shot_splits_kge(g,directory,etype,K,val_pct=0.005) :
    if not os.patorch.exists(directory+'train'+str(K)+'.txt'):
        num_nodes_per_types = {}
        for ntype in g.ntypes:
            num_nodes_per_types[ntype] = g.number_of_nodes(ntype)

        train_edges = {}
        valid_edges = {}
        test_edges = {}
        valid_edgesfgraph = {}
        test_edgesfgraph = {}
        for c_etype in g.canonical_etypes:
            etyp_eids = g.all_edges(form='uv', etype=c_etype)
            n_edges = etyp_eids[0].size(0)
            perm = torch.randperm(n_edges)
            if c_etype[1] not in etype:
                train_id = perm#[:int(n_edges * train_pct)]
                val_id = []#perm[int(n_edges * train_pct):int(n_edges * (train_pct + val_pct))]
                test_id = []#perm[int(n_edges * (train_pct + val_pct)):]
            else:
                train_id = perm[:K]
                val_id = perm[K:K + int(val_pct * len(etyp_eids[0]))]
                test_id = perm[K + int(val_pct * len(etyp_eids[0])):]

            edges = list(tuple(zip(etyp_eids[0].cpu().numpy(), etyp_eids[1].cpu().numpy())))
            train_edges[c_etype] = [edges[i] for i in train_id.numpy().astype(int)]
            if len(val_id)>0:
                valid_edges[c_etype] = [edges[i] for i in val_id.numpy().astype(int)]
            if len(test_id) > 0:
                test_edges[c_etype] = [edges[i] for i in test_id.numpy().astype(int)]
        def totriple(edges):
            triples=[]
            for e in edges.keys():
                triples+=[(uv[0],e,uv[1])  for uv in edges[e]]
            return triples
        train_triples=totriple(train_edges)
        valid_triples = totriple(valid_edges)
        test_triples = totriple(test_edges)
        with open(directory+'train'+str(K)+'.txt', 'w') as f:
            for item in train_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'valid'+str(K)+'.txt', 'w') as f:
            for item in valid_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
        with open(directory+'test'+str(K)+'.txt', 'w') as f:
            for item in test_triples:
                f.writelines("{}\t{}\t{}\n".format(item[0], item[1], item[2]))
    return


def create_edge_graph_splits(g, train_pct, val_pct, directory):
    if train_pct==1 and os.patorch.exists(directory+ "complete_splits_dir.pickle"):
        splits_dir = pickle.load(open(os.patorch.join(directory,"complete_splits_dir.pickle"), "rb"))

        train_g = splits_dir['train_g']
        valid_g = splits_dir['valid_g']
        test_g = splits_dir['test_g']
        train_edges = splits_dir['train_edges']
        valid_edges = splits_dir['valid_edges']
        test_edges = splits_dir['test_edges']

        return train_g, valid_g, test_g, train_edges, valid_edges, test_edges
    elif not os.patorch.exists(directory+"splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"):
        num_nodes_per_types = {}
        for ntype in g.ntypes:
            num_nodes_per_types[ntype] = g.number_of_nodes(ntype)

        train_edges = {}
        valid_edges = {}
        test_edges = {}
        valid_edgesfgraph = {}
        test_edgesfgraph = {}
        for c_etype in g.canonical_etypes:
            etyp_eids = g.all_edges(form='uv', etype=c_etype)
            n_edges = etyp_eids[0].size(0)
            perm = torch.randperm(n_edges)
            train_id = perm[:int(n_edges * train_pct)]
            val_id = perm[int(n_edges * train_pct):int(n_edges * (train_pct + val_pct))]
            val_id_fgraph = perm[:int(n_edges * (train_pct + val_pct))]
            test_id = perm[int(n_edges * (train_pct + val_pct)):]
            test_id_fgraph = perm
            edges = list(tuple(zip(etyp_eids[0].cpu().numpy(), etyp_eids[1].cpu().numpy())))
            train_edges[c_etype] = [edges[i] for i in train_id.numpy().astype(int)]
            valid_edges[c_etype] = [edges[i] for i in val_id.numpy().astype(int)]
            valid_edgesfgraph[c_etype] = [edges[i] for i in val_id_fgraph.numpy().astype(int)]
            test_edges[c_etype] = [edges[i] for i in test_id.numpy().astype(int)]
            test_edgesfgraph[c_etype] = [edges[i] for i in test_id_fgraph.numpy().astype(int)]


        train_g = dgl.heterograph(train_edges, num_nodes_per_types)
        valid_g = dgl.heterograph(valid_edgesfgraph, num_nodes_per_types)
        test_g = dgl.heterograph(test_edgesfgraph, num_nodes_per_types)
        for e in train_edges.keys():
            train_edges[e] = torch.tensor(train_edges[e]).long().transpose(1, 0)
        if train_pct != 1:
            for e in valid_edges.keys():
                valid_edges[e] = torch.tensor(valid_edges[e]).long().transpose(1, 0)
            for e in test_edges.keys():
                test_edges[e] = torch.tensor(test_edges[e]).long().transpose(1, 0)
        for ntype in g.ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                train_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
                valid_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
                test_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['h_f']
        splits_dir = {"train_g": train_g, "valid_g": valid_g, "test_g": test_g, "train_edges": train_edges,
                      "valid_edges": valid_edges, "test_edges": test_edges, }
        if train_pct==1:
            splits_dir = {"train_g": g, "valid_g": g, "test_g": g, "train_edges": train_edges,
                          "valid_edges": valid_edges, "test_edges": test_edges, }
            pickle.dump(splits_dir, open(os.patorch.join(directory, "complete_splits_dir.pickle"), "wb"),
                        protocol=4);
        else:
            pickle.dump(splits_dir, open(os.patorch.join(directory, "splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"), "wb"),
                    protocol=4);
    else:
        splits_dir = pickle.load(open(os.patorch.join(directory,
                    "splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"), "rb"))

        train_g = splits_dir['train_g']
        valid_g = splits_dir['valid_g']
        test_g = splits_dir['test_g']
        train_edges = splits_dir['train_edges']
        valid_edges = splits_dir['valid_edges']
        test_edges = splits_dir['test_edges']

    return train_g, valid_g, test_g, train_edges, valid_edges, test_edges


def keep_frequent_motifs(g):
    # keeps columns where the number of nonzero is more than 10% of the nodes
    for ntype in g.ntypes:
        num_motifs = g.nodes[ntype].data['motifs'].shape[1]
        num_nodes = g.nodes[ntype].data['motifs'].shape[0]
        to_keep_inds = []
        for i in range(num_motifs):
            nnz = len(torch.nonzero(g.nodes[ntype].data['motifs'][:, i]))
            if nnz > num_nodes / 10:
                to_keep_inds += [i]
        print('Motifs to keep')
        print(to_keep_inds)
        g.nodes[ntype].data['motifs'] = g.nodes[ntype].data['motifs'][:, to_keep_inds]
    return g
def motif_distribution_to_clusters(g,cluster_number):
    for ntype in g.ntypes:
        g.nodes[ntype].data['motifs']=compute_cluster_assignemnts(g.nodes[ntype].data['motifs'], cluster_number)
    return g
def motif_distribution_to_zero_one(g,args):
    if args.motif_clusters>0:
        g=motif_distribution_to_clusters(g, args.motif_clusters)
    else:
        g=motif_distribution_to_high_low_one(g)
    return g


def motif_distribution_to_high_low_one(g):
    # convert the motif distribution to high (1) and low (0) values
    med=False
    mean=True
    for ntype in g.ntypes:
        num_motifs = g.nodes[ntype].data['motifs'].shape[1]
        for i in range(num_motifs):
            if med==True:
                med = median(g.nodes[ntype].data['motifs'][:, i])
            elif mean:
                med = torch.mean(g.nodes[ntype].data['motifs'][:, i])
            else:
                med=0
            g.nodes[ntype].data['motifs'][:, i]=(g.nodes[ntype].data['motifs'][:, i]>med).float()
            print('Median motif value')
            print(med)
    # TODO possibly filter out again the frequent nonzero columns
    #  g=keep_frequent_motifs(g)
    return g

def load_link_pred_wn_pick_data(args):
    data_folder = "../data/kg/wn18/"

    # In[13]:

    data = pickle.load(open(os.patorch.join(data_folder, 'data_lp_motifs.pickle'), "rb"))
    train_edges=data["train_edges"]
    test_edges=data["test_edges"]
    valid_edges=data["valid_edges"]
    train_g=data["train_g"]
    valid_g = data["valid_g"]
    test_g=data["test_g"]
    featless_node_types=data["featless_node_types"]
    src_id=data["src_id"]
    dest_id=data["dest_id"]
    edata=data["edata"]
    if args.use_node_motifs:
        for ntype in train_g.ntypes:
            train_g.nodes[ntype].data['motifs'] = train_g.nodes[ntype].data['motifs'].float()
        train_g=keep_frequent_motifs(train_g)
        train_g=motif_distribution_to_zero_one(train_g,args)
    else:
        for ntype in train_g.ntypes:
            del train_g.nodes[ntype].data['motifs']


    return train_edges, test_edges, valid_edges, train_g,valid_g,test_g, featless_node_types

def load_link_pred_wn_data(args):
    def triplets_to_dict(edges,etype_to_canonical):
        d_e={}
        s,e,d=edges
        for sou,edg,dest in zip(s,e,d):
            edg =str(edg)
            edg=etype_to_canonical[edg]
            if edg not in d_e:
                d_e[edg]=[(sou,dest)]
            else:
                d_e[edg]+=[(sou,dest)]
        return d_e
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../../data/kg/wn18/"

    # In[13]:

    g = pickle.load(open(os.patorch.join(data_folder, 'graph_reduced.pickle'), "rb")).to(torch.device("cpu"))
    link_pred_splits=pickle.load(open(os.patorch.join(data_folder, 'link_pred_splits.pickle'), "rb"))#.to(torch.device("cpu"))
    num_nodes_per_types={}
    for ntype in g.ntypes:
        num_nodes_per_types[ntype]=g.number_of_nodes(ntype)
    # In[14]:
    etype_to_canonical={}
    for i, etype in enumerate(g.etypes):
        etype_to_canonical[etype]=g.canonical_etypes[i]

    train_edges=triplets_to_dict(link_pred_splits['tr'],etype_to_canonical)
    test_edges = triplets_to_dict(link_pred_splits['test'],etype_to_canonical)
    valid_edges =triplets_to_dict( link_pred_splits['val'],etype_to_canonical)
    train_g=dgl.heterograph(train_edges,num_nodes_per_types)
    valid_g = dgl.heterograph(valid_edges, num_nodes_per_types)
    # TODO THIS IS WRONG!!! I have to add the train valid and test
    test_g = 1 #dgl.heterograph(test_edges, num_nodes_per_types)
    # remove last feature
    g.nodes[ntype].data['features']=g.nodes[ntype].data['features'][:,:-1]
    use_feats=True
    if use_feats:
        for ntype in g.ntypes:
            if g.nodes[ntype].data.get("features", None) is not None:
                train_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['features']
                valid_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['features']
                test_g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['features']
    # Create the train, valid, test graphs
    for e in train_edges.keys():
        train_edges[e]=torch.tensor(train_edges[e]).long().transpose(1,0)
    for e in valid_edges.keys():
        valid_edges[e]=torch.tensor(valid_edges[e]).long().transpose(1,0)
    for e in test_edges.keys():
        test_edges[e]=torch.tensor(test_edges[e]).long().transpose(1,0)

    featless_node_types=[]

    return train_edges, test_edges, valid_edges, train_g,valid_g,test_g, featless_node_types

def load_wn_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/kg/wn18/"

    # In[13]:
    # graph file has 81 different node types based on the type of word (but it is unclear what it corresponds to)
    # graph_reduced has the 4 basic node types.
    g = pickle.load(open(os.patorch.join(data_folder, 'graph_reduced.pickle'), "rb")).to(torch.device("cpu"))

    # In[14]:
    labels = g.nodes['word'].data['features'][:, -1].cpu()
    g.nodes['word'].data['features']=g.nodes['word'].data['features'][:,: -1]

    label_indices = [i for i in range(len(labels))];
    train_idx, test_idx, y_train, y_test = train_test_split(label_indices, labels, test_size=0.2, random_state=seed)

    #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed);
    #train_idx, test_idx = next(sss.split(label_indices, labels));
    val_idx, test_idx, y_train, y_test = train_test_split(list(test_idx), np.array(labels)[test_idx], test_size=0.5, random_state=seed)
    #sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed);
    #valid_index_temp, test_index_temp = next(sss.split(list(test_idx), np.array(labels)[test_idx]));
    #val_idx = np.array(test_idx)[valid_index_temp];
    #test_idx = np.array(test_idx)[test_index_temp];

    train_idx = np.array(train_idx);
    test_idx = np.array(test_idx);
    val_idx = np.array(val_idx);
    category='word'
    num_classes=4
    for ntype in g.ntypes:
        if g.nodes[ntype].data.get("features", None) is not None:
            g.nodes[ntype].data['h_f'] = g.nodes[ntype].data['features']

    featless_node_types = []
    if num_classes>1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i,int(labels[i])]=1
    else:
        labels_n=labels
    labels=labels_n
    if args.use_clusterandrecover_loss:
        for ntype in g.ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                g.nodes[ntype].data['h_clusters']=compute_cluster_assignemnts(g.nodes[ntype].data['h_f'],cluster_number=args.num_clusters)

    return train_idx,test_idx,val_idx,labels,g,category,num_classes,featless_node_types
def load_kaggle_shoppers_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/kaggle_shoppers/"

    # In[13]:r

    G = pickle.load(open(os.patorch.join(data_folder, 'graph_0.001.pickle'), "rb")).to(torch.device("cpu"))

    # In[14]:

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    # In[15]:

    print(G)

    # In[16]:

    # G.nodes['application'].data['features'].fill_(0.0);

    # In[17]:

    print(labels)

    # In[18]:

    label_indices = [i for i in range(len(labels))];
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed);
    train_idx, test_idx = next(sss.split(label_indices, labels));

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed);
    valid_index_temp, test_index_temp = next(sss.split(list(test_idx), np.array(labels)[test_idx]));
    val_idx = np.array(test_idx)[valid_index_temp];
    test_idx = np.array(test_idx)[test_index_temp];

    train_idx = np.array(train_idx);
    test_idx = np.array(test_idx);
    val_idx = np.array(val_idx);
    for ntype in G.ntypes:
        if G.nodes[ntype].data.get("features", None) is not None:
            G.nodes[ntype].data['h_f'] = G.nodes[ntype].data['features']
    category='history'
    num_classes=1
    featless_node_types = ['brand', 'customer', 'chain', 'market', 'dept', 'category', 'company']
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types
def load_imdb_prexiang_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_data/xiang/"

    # In[13]:
    # load to cpu for very large graphs
    file='dgl-neptune-dataset.pickle'

    dataset=pickle.load(open(os.patorch.join(data_folder, file), "rb"))
    G=dataset.g

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G=keep_frequent_motifs(G)
        G=motif_distribution_to_zero_one(G,args)
        print(sum(G.nodes[ntype].data['motifs']))
    for ntype in dataset.features.keys():
        G.nodes[ntype].data["h_f"]=dataset.features[ntype]
    category = 'title'
    train_idx, train_label = dataset.train_set[category]
    val_idx, val_label = dataset.valid_set[category]
    test_idx, test_label = dataset.test_set[category]
    num_classes = len(list(dataset.labels.values())[0].label_map)
    labels = torch.zeros((G.number_of_nodes(category), len(list(dataset.labels.values())[0].label_map)))
    labels[train_idx] = train_label.float()
    labels[val_idx] = val_label.float()
    labels[test_idx] = test_label.float()

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)

    featless_node_types = []
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters']=compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],cluster_number=args.num_clusters)
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types
def create_label_split(num_nodes,train_pct,val_pct=0.05):
    tot=list(np.arange(num_nodes))
    random.shuffle(tot)
    train_idx=tot[:int(num_nodes*train_pct)]
    val_idx = tot[int(num_nodes * train_pct):int(num_nodes * train_pct)+int(num_nodes * val_pct)]
    test_idx = tot[int(num_nodes * train_pct)+int(num_nodes * val_pct):]
    return (train_idx),(val_idx),(test_idx)
def load_imdb_kge_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)

    if args.few_shot:
        create_edge_graph_few_shot_splits_kge(G,data_folder,etype=['Drama_directed_by','directed_Drama'], K=args.k_shot_edge)
    else:
        create_edge_graph_splits_kge(G, 0.975-args.test_edge_split, 0.025,data_folder)
    print(G)
    return
def load_oag_kge_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:


    # In[13]:
    # load to cpu for very large graphs
    data_folder = "../data/oag/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.patorch.join(data_folder, 'graph.pickle'), "rb"))

    create_edge_graph_splits_kge(G, 0.975-args.test_edge_split, 0.025,data_folder)
    print(G)
    return
def load_oag_na_kge_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:


    # In[13]:
    # load to cpu for very large graphs
    data_folder = "../data/oag_na/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.patorch.join(data_folder, 'graph_na.pickle'), "rb"))

    create_edge_graph_splits_kge(G, 0.975-args.test_edge_split, 0.025,data_folder)
    print(G)
    return

def load_dblp_kge_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/dblp_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)

    if args.few_shot:
        create_edge_graph_few_shot_splits_kge(G,data_folder,etype=['writted_by_3','3_writes'], K=args.k_shot_edge)
    else:
        create_edge_graph_splits_kge(G, 0.975-args.test_edge_split, 0.025,data_folder)
    print(G)
    return

def load_dblp_few_edge_shot_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/dblp_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,
                                                                                                 etype=['writted_by_3','3_writes'],K=args.k_shot_edge)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['actor'] = ['played', 'played_by'] * 2
        metapaths['director'] = ['directed', 'directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    if args.splitpct is not None:
        if args.splitpct==0.1:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
            train_idx = train_val_test_idx['train_idx']
            val_idx = train_val_test_idx['val_idx']
            test_idx = train_val_test_idx['test_idx']
        else:
            train_idx,val_idx,test_idx=create_label_split(labels.shape[0],args.splitpct)

    else:
        if args.k_fold > 0:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
        else:
            if args.split == 5:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
            else:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']
    print(G)

    print(labels)



    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'author'
    num_classes = 4
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = ['conference']
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def load_imdb_few_edge_shot_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]

    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,
                                                                                                 etype=['Drama_directed_by','directed_Drama'],K=args.k_shot_edge)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['actor'] = ['played', 'played_by'] * 2
        metapaths['director'] = ['directed', 'directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    if args.splitpct is not None:
        if args.splitpct==0.1:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
            train_idx = train_val_test_idx['train_idx']
            val_idx = train_val_test_idx['val_idx']
            test_idx = train_val_test_idx['test_idx']
        else:
            train_idx,val_idx,test_idx=create_label_split(labels.shape[0],args.splitpct)

    else:
        if args.k_fold > 0:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
        else:
            if args.split == 5:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
            else:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']
    print(G)

    print(labels)



    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'movie'
    num_classes = 3
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = []
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g
def load_oag_univ_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/oag/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.patorch.join(data_folder, 'graph.pickle'), "rb"))
    for ntype in G.ntypes:
        if G.nodes[ntype].data.get("emb", None) is not None:
            G.nodes[ntype].data['h_f'] =  G.nodes[ntype].data['emb']
    if args.few_shot:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,etype=['Drama_directed_by','directed_Drama'], K=args.k_shot_edge)
    else:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.975-args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['actor'] = ['played', 'played_by'] * 2
        metapaths['director'] = ['directed', 'directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))
    labels=labels.todense()
    if args.splitpct is not None:
        train_idx,val_idx,test_idx=create_label_split(labels.shape[0],args.splitpct)

    else:
        if args.k_fold > 0:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
        else:
            if args.split == 5:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
            else:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']
    print(G)

    print(labels)



    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'paper'
    num_classes = 5

    labels = torch.tensor(labels)
    featless_node_types = ['author']
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def load_oag_na_univ_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/oag_na/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.patorch.join(data_folder, 'graph_na.pickle'), "rb"))
    for ntype in G.ntypes:
        if G.nodes[ntype].data.get("emb", None) is not None:
            G.nodes[ntype].data['h_f'] =  G.nodes[ntype].data['emb']
    if args.few_shot:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,etype=['Drama_directed_by','directed_Drama'], K=args.k_shot_edge)
    else:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.975-args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['actor'] = ['played', 'played_by'] * 2
        metapaths['director'] = ['directed', 'directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))
    labels=labels.todense()
    if args.splitpct is not None:
        train_idx,val_idx,test_idx=create_label_split(labels.shape[0],args.splitpct)

    else:
        if args.k_fold > 0:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
        else:
            if args.split == 5:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
            else:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']
    print(G)

    print(labels)



    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'paper'
    num_classes = 5

    labels = torch.tensor(labels)
    featless_node_types = ['author']
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def load_oag_full_univ_data(args):
    #OAGData=OAGDataset.load()
    #OAGData
    return
def load_std_het_full_univ_data(args):
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()
    g = dataset[0]
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = g.nodes[category].data.pop('train_mask')
    test_mask = g.nodes[category].data.pop('test_mask')
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()
    val_idx = train_idx
    labels = g.nodes[category].data.pop('labels')
    G = g
    data_folder = "../data/"+args.dataset+"/"
    metapaths = {}
    if args.rw_supervision:
        '''
            TODO add metapaths
        '''
    use_default_split = True
    if not use_default_split:
        train_idx, val_idx, test_idx = create_label_split(labels.shape[0], args.splitpct, val_pct=0.00801)
    print(G)
    featless_node_types = []

    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G,
                                                                                              0.975 - args.test_edge_split,
                                                                                              0.025,
                                                                                              data_folder)

    return train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, metapaths, \
           train_edges, test_edges, valid_edges, train_g, valid_g, test_g


def load_ogbn_mag_full_univ_data(args):
        use_cuda = args.gpu
        check_cuda = torch.cuda.is_available()
        if use_cuda < 0:
            check_cuda = False;
        device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
        print("Using device", device)
        cpu_device = torch.device("cpu");
        dataset = DglNodePropPredDataset(name=args.dataset)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))
        #node_feats=[]
        for ntype in hg.ntypes:
                if len(hg.nodes[ntype].data) == 0:
                    x=0
                    #node_feats.append(None)
                else:
                    assert len(hg.nodes[ntype].data) == 1
                    feat = hg.nodes[ntype].data.pop('feat')
                    hg.nodes[ntype].data['h_f']=feat
                    #node_feats.append(feat.share_memory_())
        data_folder = "../data/ogbn-mag/"
        G=hg
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(hg,
                                                                                                  0.975 - args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)
        if args.use_node_motifs:
            '''
            TODO add motifs
            '''

        metapaths = {}
        if args.rw_supervision:
            '''
                TODO add metapaths
            '''
        use_default_split=True
        if not use_default_split:
            train_idx, val_idx, test_idx = create_label_split(labels.shape[0], args.splitpct, val_pct=0.00801)
        print(G)

        print(labels)

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        val_idx = np.array(val_idx)
        num_classes = 349
        multilabel=False
        if multilabel:
            labels_n = torch.zeros((np.shape(labels)[0], num_classes))

            for i in range(np.shape(labels)[0]):
                labels_n[i, int(labels[i]) if int(labels[i]) < 6 else int(labels[i]) - 1] = 1
        else:
            labels_n = torch.tensor(labels)

        labels = labels_n
        featless_node_types = []

        if args.use_clusterandrecover_loss:
            for ntype in G.ntypes:
                if G.nodes[ntype].data.get("h_f", None) is not None:
                    G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                    cluster_number=args.num_clusters)
                    train_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                    valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                    test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
        return train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, metapaths, \
               train_edges, test_edges, valid_edges, train_g, valid_g, test_g



def load_acm_univ_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)

    data_folder = '../data/acm/'
    data_file_path = '../data/acm/ACM.mat'

    data = scipy.io.loadmat(data_file_path)

    G = dgl.heterograph({
        ('paper', 'written-by', 'author'): data['PvsA'].nonzero(),
        ('author', 'writing', 'paper'): data['PvsA'].transpose().nonzero(),
        ('paper', 'citing', 'paper'): data['PvsP'].nonzero(),
        ('paper', 'cited', 'paper'): data['PvsP'].transpose().nonzero(),
        ('paper', 'is-about', 'subject'): data['PvsL'].nonzero(),
        ('subject', 'has', 'paper'): data['PvsL'].transpose().nonzero(),
    })
    print(G)

    pvc = data['PvsC'].tocsr()
    p_selected = pvc.tocoo()
    # generate labels
    labels = pvc.indices
    labels = torch.tensor(labels).long()

    # generate train/val/test split
    pid = p_selected.row
    shuffle = np.random.permutation(pid)
    train_idx = torch.tensor(shuffle[0:800]).long()
    val_idx = torch.tensor(shuffle[800:900]).long()
    test_idx = torch.tensor(shuffle[900:]).long()

    for ntype in G.ntypes:
        emb = torch.nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), 256), requires_grad=False)
        torch.nn.init.xavier_uniform_(emb)
        G.nodes[ntype].data['h_f'] = emb

    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G,
                                                                                              0.975 - args.test_edge_split,
                                                                                              0.025,
                                                                                              data_folder)
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']

    metapaths = {}
    if args.rw_supervision:
        '''
            TODO add metapaths
        '''


    train_idx, val_idx, test_idx = create_label_split(labels.shape[0], args.splitpct,val_pct=0.00801)
    print(G)

    print(labels)

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'paper'
    num_classes = 13
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i]) if int(labels[i]) < 6 else int(labels[i])-1] = 1
    else:
        labels_n = labels

    labels = labels_n
    num_classes=13
    featless_node_types = []

    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, metapaths, \
           train_edges, test_edges, valid_edges, train_g, valid_g, test_g


def load_imdb_univ_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    #if args.few_shot:
    #    edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
    #    G = dgl.heterograph(edge_list)
    edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    #if args.few_shot:
    #    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,etype=['Drama_directed_by','directed_Drama'], K=args.k_shot_edge)
    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.975-args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['actor'] = ['played', 'played_by'] * 2
        metapaths['director'] = ['directed', 'directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    if args.splitpct is not None:
        if args.splitpct==0.1:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
            train_idx = train_val_test_idx['train_idx']
            val_idx = train_val_test_idx['val_idx']
            test_idx = train_val_test_idx['test_idx']
        else:
            train_idx,val_idx,test_idx=create_label_split(labels.shape[0],args.splitpct)

    else:
        if args.k_fold > 0:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
        else:
            if args.split == 5:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
            else:
                train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
        train_idx = train_val_test_idx['train_idx']
        val_idx = train_val_test_idx['val_idx']
        test_idx = train_val_test_idx['test_idx']
    print(G)

    print(labels)



    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'movie'
    num_classes = 3
    multilabel=False
    if multilabel:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = torch.tensor(labels)
    labels = labels_n
    featless_node_types = []
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g
def load_dblp_univ_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/dblp_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    if args.few_shot:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    if args.few_shot:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G, data_folder,
                                                                                                     etype=[
                                                                                                         'writted_by_3','3_writes'],
                                                                                                     K=args.k_shot_edge)
    else:
        if args.test_edge_split==0:
            train_g=G
            valid_g=G
            test_g=G
            train_edges=edge_list
            valid_edges=edge_list
            test_edges=edge_list
        else:
            train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G,
                                                                                                  0.975 - args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        multiplicity = 1
        metapaths['paper'] = ['writted_by', 'writes'] * multiplicity
        metapaths['conference'] = ['includes', 'writted_by', 'writes', 'prereseted_in'] * multiplicity
        metapaths['author'] = ['writes','contains','contained_by', 'writted_by'] * multiplicity

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    #train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-' + str(args.k_fold) + '.npz')
    train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')
    if args.k_fold > 0:
        raise NotImplementedError

    print(G)

    print(labels)

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category = 'author'
    num_classes = 4
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = ['conference']
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def re_e_list(edge_list,folder):
    n_edge_list={}
    for k in edge_list.keys():
        nk=(k[0],k[0]+"_"+k[1]+"_"+k[2],k[2])
        n_edge_list[nk]=edge_list[k]
    pickle.dump(n_edge_list, open(os.patorch.join(folder, "edge_list.pickle"), "wb"),
                protocol=4);
    return n_edge_list
def load_drkg_univ_data(args):
    def create_dgl_hetero_from_triplets(triplets):
        entity_dictionary = {}

        def insert_entry(entry, ent_type, dic):
            if ent_type not in dic:
                dic[ent_type] = {}
            ent_n_id = len(dic[ent_type])
            if entry not in dic[ent_type]:
                dic[ent_type][entry] = ent_n_id
            return dic

        for triple in triplets:
            src = triple[0]
            split_src = src.split('::')
            src_type = split_src[0]
            dest = triple[2]
            split_dest = dest.split('::')
            dest_type = split_dest[0]
            insert_entry(src, src_type, entity_dictionary)
            insert_entry(dest, dest_type, entity_dictionary)

        edge_dictionary = {}
        for triple in triplets:
            src = triple[0]
            split_src = src.split('::')
            src_type = split_src[0]
            dest = triple[2]
            split_dest = dest.split('::')
            dest_type = split_dest[0]

            src_int_id = entity_dictionary[src_type][src]
            dest_int_id = entity_dictionary[dest_type][dest]

            pair = (src_int_id, dest_int_id)
            etype = (src_type, triple[1], dest_type)
            if etype in edge_dictionary:
                edge_dictionary[etype] += [pair]
            else:
                edge_dictionary[etype] = [pair]

        graph = dgl.heterograph(edge_dictionary)
        return graph
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/drkg/drkg/"


    #df = pd.read_csv(data_folder+'drkg.tsv', sep="\t", header=None)
    #triplets = df.values.tolist()
    #G = create_dgl_hetero_from_triplets(triplets)

    #train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.95, 0.025,
    #                                                                                              data_folder)

    splits_dir=pickle.load(open(os.patorch.join(data_folder, 'splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['Gene'] = ['DGIDB::INHIBITOR::Gene:Compound','DRUGBANK::treats::Compound:Disease'] * 1
        metapaths['Compound'] = ['DRUGBANK::treats::Compound:Disease','Hetionet::DdG::Disease:Gene'] * 1
        #metapaths['function'] = ['0', 'played'] * 1


    print(test_g)


    train_idx = None
    val_idx = None
    test_idx = None
    category = None
    num_classes = None
    labels= None
    featless_node_types = G.ntypes

    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g
def load_drkg_edge_few_shot_data(args):
    def create_dgl_hetero_from_triplets(triplets):
        entity_dictionary = {}

        def insert_entry(entry, ent_type, dic):
            if ent_type not in dic:
                dic[ent_type] = {}
            ent_n_id = len(dic[ent_type])
            if entry not in dic[ent_type]:
                dic[ent_type][entry] = ent_n_id
            return dic

        for triple in triplets:
            src = triple[0]
            split_src = src.split('::')
            src_type = split_src[0]
            dest = triple[2]
            split_dest = dest.split('::')
            dest_type = split_dest[0]
            insert_entry(src, src_type, entity_dictionary)
            insert_entry(dest, dest_type, entity_dictionary)

        edge_dictionary = {}
        for triple in triplets:
            src = triple[0]
            split_src = src.split('::')
            src_type = split_src[0]
            dest = triple[2]
            split_dest = dest.split('::')
            dest_type = split_dest[0]

            src_int_id = entity_dictionary[src_type][src]
            dest_int_id = entity_dictionary[dest_type][dest]

            pair = (src_int_id, dest_int_id)
            etype = (src_type, triple[1], dest_type)
            if etype in edge_dictionary:
                edge_dictionary[etype] += [pair]
            else:
                edge_dictionary[etype] = [pair]
        pickle.dump(entity_dictionary, open(os.patorch.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "wb"),
                    protocol=4);
        graph = dgl.heterograph(edge_dictionary)
        return graph
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/drkg/drkg/"

    '''
    df = pd.read_csv(data_folder+'drkg.tsv', sep="\t", header=None)
    triplets = df.values.tolist()
    G = create_dgl_hetero_from_triplets(triplets)

    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 1, 0,
                                                                                                  data_folder)
    '''
    splits_dir=pickle.load(open(os.patorch.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['Gene'] = ['DGIDB::INHIBITOR::Gene:Compound','DRUGBANK::treats::Compound:Disease'] * 1
        metapaths['Compound'] = ['DRUGBANK::treats::Compound:Disease','Hetionet::DdG::Disease:Gene'] * 1
        #metapaths['function'] = ['0', 'played'] * 1


    print(test_g)


    train_idx = None
    val_idx = None
    test_idx = None
    category = None
    num_classes = None
    labels= None
    featless_node_types = G.ntypes

    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def load_query_biodata_univ_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/query_biodata/"

    # In[13]:
    #edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
    #G = dgl.heterograph(edge_list)

    #train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.95, 0.025,
    #                                                                                              data_folder)

    splits_dir=pickle.load(open(os.patorch.join(data_folder, 'splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G, args)
        for ntype in G.ntypes:
            train_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            valid_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
            test_g.nodes[ntype].data['motifs'] = G.nodes[ntype].data['motifs']
    metapaths = {}
    if args.rw_supervision:
        metapaths['drug'] = ['drug_sexual_disorder_drug', 'drug_sleep_disorder_drug'] * 1
        metapaths['protein'] = ['protein_activation_protein', 'protein_activation_protein'] * 1
        #metapaths['function'] = ['0', 'played'] * 1


    print(test_g)


    train_idx = None
    val_idx = None
    test_idx = None
    category = None
    num_classes = None
    labels= None
    featless_node_types = G.ntypes

    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g

def load_imdb_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    edge_list = pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)

    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G=keep_frequent_motifs(G)
        G=motif_distribution_to_zero_one(G,args)

    metapaths = {}
    if args.rw_supervision is not None and args.rw_supervision :
        metapaths['actor'] = ['played',  'played_by'] * 2
        metapaths['director'] = ['directed','directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))

    if args.k_fold>0:
        train_val_test_idx = np.load(data_folder + 'train_val_test_idx_kfold-'+str(args.k_fold)+'.npz')
    else:
        if args.split==5:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx005.npz')
        else:
            train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')


    print(G)


    print(labels)

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category='movie'
    num_classes = 3
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))

        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = []
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters']=compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],cluster_number=args.num_clusters)
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types,metapaths

def load_dblp_preprocessed_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/dblp_preprocessed/"

    # In[13]:
    # load to cpu for very large graphs
    edge_list=pickle.load(open(os.patorch.join(data_folder, 'edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.patorch.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] =features[ntype]

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.patorch.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G,args)
    labels = pickle.load(open(os.patorch.join(data_folder, 'labels.pickle'), "rb"))
    train_val_test_idx = np.load(data_folder + 'train_val_test_idx.npz')

    print(G)
    metapaths = {}
    if args.rw_supervision:
        multiplicity=1
        metapaths['paper'] = ['writted_by',  'writes'] * multiplicity
        metapaths['conference'] = ['includes','writted_by',  'writes','prereseted_in'] * multiplicity
        metapaths['author'] = ['writes', 'writted_by'] * multiplicity

    print(labels)

    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category='author'
    num_classes = 4
    if num_classes > 1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))
        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = ['conference']
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters']=compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],cluster_number=args.num_clusters)
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types,metapaths

def load_imdb_data(args):
    use_cuda = args.gpu
    check_cuda = torch.cuda.is_available()
    if use_cuda < 0:
        check_cuda = False;
    device = torch.device("cuda:" + str(use_cuda) if check_cuda else "cpu")
    print("Using device", device)
    cpu_device = torch.device("cpu");

    # In[10]:

    seed = 0;
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

    # In[12]:

    data_folder = "../data/imdb_data/"

    # In[13]:
    # load to cpu for very large graphs
    G = pickle.load(open(os.patorch.join(data_folder, 'graph_red.pickle'), "rb")).to(torch.device("cpu"))

    # extract adult label from graph
    label_type='genre'
    if label_type=='adult':
        labels = G.nodes['movie'].data['features'][:, 602]
        G.nodes['movie'].data['features'] = torch.cat(
        (G.nodes['movie'].data['features'][:, :602], G.nodes['movie'].data['features'][:, 603:]), 1)
    elif label_type=='genre':
        # last 30 are the genre labels
        #CHECK HOW MANY ARE THE GENRE LABELS MAYBE 28
        labels = G.nodes['movie'].data['features'][:, -30:]
        # Discard very rare classes
        s_labels=sum(labels)
        filt_nbr=40
        filter_labels=s_labels>=filt_nbr
        labels = labels[:, filter_labels]

        G.nodes['movie'].data['features'] = (G.nodes['movie'].data['features'][:, :-30])
    else:
        raise NotImplementedError


    # In[15]:
    G.nodes['person'].data['features'] = G.nodes['person'].data['features'].float()
    G.nodes['movie'].data['features'] = G.nodes['movie'].data['features'].float()
    labels=labels.float().cpu()
    print(G)

    # In[16]:

    # G.nodes['application'].data['features'].fill_(0.0);

    # In[17]:

    print(labels)

    # In[18]:

    label_indices = [i for i in range(len(labels))]
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(msss.split(label_indices, labels));

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed);
    valid_index_temp, test_index_temp = next(msss.split(list(test_idx), np.array(labels)[test_idx]));
    val_idx = np.array(test_idx)[valid_index_temp]
    test_idx = np.array(test_idx)[test_index_temp]

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    val_idx = np.array(val_idx)
    category='movie'
    for ntype in G.ntypes:
        if G.nodes[ntype].data.get("features", None) is not None:
            G.nodes[ntype].data['h_f'] = G.nodes[ntype].data['features']
    num_classes=labels.shape[1]
    featless_node_types = []
    if args.use_clusterandrecover_loss:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters']=compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],cluster_number=args.num_clusters)
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types


def load_gen_data(args):
    data = load_data(args.dataset, bfs_level=args.bfs_level, relabel=args.relabel)
    num_nodes = data.num_nodes
    num_rels = data.num_rels
    num_classes = data.num_classes
    labels = data.labels
    train_idx = data.train_idx
    test_idx = data.test_idx

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # since the nodes are featureless, the input feature is then the node id.
    feats = torch.arange(num_nodes)
    return num_nodes,num_rels,num_classes,train_idx,test_idx,val_idx,labels,feats,data.edge_type,data.edge_norm,data.edge_src,data.edge_dst