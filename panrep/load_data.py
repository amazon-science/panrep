'''
This file contains functions that help loading the different datasets in the required format.
'''

import os
import pickle
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import dgl
import numpy as np
import torch
from dgl.contrib.data import load_data
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split
from statistics import median
from scipy.cluster.vq import vq, kmeans2, whiten
import pandas as pd
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
    if args.dataset == "imdb_preprocessed":
        train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
            train_edges, test_edges, valid_edges, train_g, valid_g, test_g= load_imdb_univ_preprocessed_data(args)
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
    elif args.dataset == "drkg":
        train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, rw_neighbors, \
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g=load_drkg_edge_few_shot_data(args)
    else:
        raise NotImplementedError
    if labels is not None and len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))
    return train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,rw_neighbors,\
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

    g = pickle.load(open(os.path.join(data_folder, 'graph.pickle'), "rb")).to(torch.device("cpu"))
    #get eid from heterograph and use dgl.edge_subgraph
    train_pct = 0.8
    val_pct= 0.1
    #train_g,valid_g,test_g,train_edges,valid_edges,test_edges=create_edge_graph_splits(g,train_pct,val_pct,data_folder)

    splits_dir=pickle.load(open(os.path.join(data_folder, 'splits_dir.pickle'), "rb"))
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
    if os.path.exists(os.path.join(directory, "few_shot_splits_dir"+str(K)+".pickle")):
        splits_dir = pickle.load(open(os.path.join(directory, "few_shot_splits_dir"+str(K)+".pickle"), "rb"))

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
        pickle.dump(splits_dir, open(os.path.join(directory, "few_shot_splits_dir"+str(K)+".pickle"), "wb"),
                    protocol=4);

    return train_g,valid_g,test_g,train_edges,valid_edges,test_edges

def create_edge_graph_splits_kge(g,train_pct,val_pct,directory):

    if not os.path.exists(directory + "splits_dir_tr" + str(train_pct) + "_val_" + str(val_pct) + ".pickle"):

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
        splits_dir = pickle.load(open(os.path.join(directory,
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
    if not os.path.exists(directory+'train'+str(K)+'.txt'):
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
    if not os.path.exists(directory+"splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"):
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
            pickle.dump(splits_dir, open(os.path.join(directory, "complete_splits_dir.pickle"), "wb"),
                        protocol=4);
        else:
            pickle.dump(splits_dir, open(os.path.join(directory, "splits_dir_tr"+str(train_pct)+"_val_"+str(val_pct)+".pickle"), "wb"),
                    protocol=4);
    else:
        splits_dir = pickle.load(open(os.path.join(directory,
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

    data = pickle.load(open(os.path.join(data_folder, 'data_lp_motifs.pickle'), "rb"))
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

    g = pickle.load(open(os.path.join(data_folder, 'graph_reduced.pickle'), "rb")).to(torch.device("cpu"))
    link_pred_splits=pickle.load(open(os.path.join(data_folder, 'link_pred_splits.pickle'), "rb"))#.to(torch.device("cpu"))
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
    g = pickle.load(open(os.path.join(data_folder, 'graph_reduced.pickle'), "rb")).to(torch.device("cpu"))

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
    if args.use_cluster:
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

    G = pickle.load(open(os.path.join(data_folder, 'graph_0.001.pickle'), "rb")).to(torch.device("cpu"))

    # In[14]:

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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

    dataset=pickle.load(open(os.path.join(data_folder, file), "rb"))
    G=dataset.g

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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
    if args.use_cluster:
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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.path.join(data_folder, 'graph.pickle'), "rb"))

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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.path.join(data_folder, 'graph_na.pickle'), "rb"))

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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
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
    edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,
                                                                                                 etype=['writted_by_3','3_writes'],K=args.k_shot_edge)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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
    if args.use_cluster:
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
    edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]

    train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,
                                                                                                 etype=['Drama_directed_by','directed_Drama'],K=args.k_shot_edge)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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
    if args.use_cluster:
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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.path.join(data_folder, 'graph.pickle'), "rb"))
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
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))
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
    if args.use_cluster:
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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        G = pickle.load(open(os.path.join(data_folder, 'graph_na.pickle'), "rb"))
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
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))
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
    if args.use_cluster:
        for ntype in G.ntypes:
            if G.nodes[ntype].data.get("h_f", None) is not None:
                G.nodes[ntype].data['h_clusters'] = compute_cluster_assignemnts(G.nodes[ntype].data['h_f'],
                                                                                cluster_number=args.num_clusters)
                train_g.nodes[ntype].data['h_clusters'] =G.nodes[ntype].data['h_clusters']
                valid_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
                test_g.nodes[ntype].data['h_clusters'] = G.nodes[ntype].data['h_clusters']
    return  train_idx,test_idx,val_idx,labels,category,num_classes,featless_node_types,metapaths,\
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
    if args.few_shot:
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    if args.few_shot:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_few_shot_splits(G,data_folder,etype=['Drama_directed_by','directed_Drama'], K=args.k_shot_edge)
    else:
        train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.975-args.test_edge_split,
                                                                                                  0.025,
                                                                                                  data_folder)

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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
    if args.use_cluster:
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
        edge_list = pickle.load(open(os.path.join(data_folder, 'k_shot_edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    else:
        edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
        G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
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
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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
    if args.use_cluster:
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
    pickle.dump(n_edge_list, open(os.path.join(folder, "edge_list.pickle"), "wb"),
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

    splits_dir=pickle.load(open(os.path.join(data_folder, 'splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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
        pickle.dump(entity_dictionary, open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "wb"),
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
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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
    #edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
    #G = dgl.heterograph(edge_list)

    #train_g, valid_g, test_g, train_edges, valid_edges, test_edges = create_edge_graph_splits(G, 0.95, 0.025,
    #                                                                                              data_folder)

    splits_dir=pickle.load(open(os.path.join(data_folder, 'splits_dir.pickle'), "rb"))

    train_g=splits_dir['train_g']
    valid_g=splits_dir['valid_g']
    test_g=splits_dir['test_g']
    train_edges=splits_dir['train_edges']
    valid_edges=splits_dir['valid_edges']
    test_edges=splits_dir['test_edges']
    G=train_g
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
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
    edge_list = pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)

    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] = features[ntype]
    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G=keep_frequent_motifs(G)
        G=motif_distribution_to_zero_one(G,args)

    metapaths = {}
    if args.rw_supervision is not None and args.rw_supervision :
        metapaths['actor'] = ['played',  'played_by'] * 2
        metapaths['director'] = ['directed','directed_by'] * 2
        metapaths['movie'] = ['played_by', 'played'] * 2

    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))

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
    if args.use_cluster:
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
    edge_list=pickle.load(open(os.path.join(data_folder, 'edge_list.pickle'), "rb"))
    G = dgl.heterograph(edge_list)
    features = pickle.load(open(os.path.join(data_folder, 'features.pickle'), "rb"))
    for ntype in features.keys():
        G.nodes[ntype].data['h_f'] =features[ntype]

    if args.use_node_motifs:
        node_motifs = pickle.load(open(os.path.join(data_folder, 'node_motifs.pickle'), "rb"))
        for ntype in G.ntypes:
            G.nodes[ntype].data['motifs'] = node_motifs[ntype].float()
        G = keep_frequent_motifs(G)
        G = motif_distribution_to_zero_one(G,args)
    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))
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
    if args.use_cluster:
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
    G = pickle.load(open(os.path.join(data_folder, 'graph_red.pickle'), "rb")).to(torch.device("cpu"))

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
    if args.use_cluster:
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