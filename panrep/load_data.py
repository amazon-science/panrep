import os
import pickle
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

import numpy as np
import torch
from dgl.contrib.data import load_data
from sklearn.model_selection import StratifiedShuffleSplit

def load_hetero_data(args):
    if args.dataset == "kaggle_shoppers":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_kaggle_shoppers_data(args)
    elif args.dataset == "wn":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_wn_data(args)
    elif args.dataset == "imdb":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_imdb_data(args)
    else:
        raise NotImplementedError
    return train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types


def load_link_pred_wn_data(args):
    def triplets_to_dict(edges):
        d_e={}
        s,e,d=edges
        for sou,edg,dest in zip(s,e,d):
            edg =str(edg)
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

    data_folder = "../data/kg/wn18/"

    # In[13]:

    g = pickle.load(open(os.path.join(data_folder, 'graph_reduced.pickle'), "rb")).to(device)
    link_pred_splits=pickle.load(open(os.path.join(data_folder, 'link_pred_splits.pickle'), "rb"))#.to(device)
    # In[14]:
    train_edges=triplets_to_dict(link_pred_splits['tr'])
    test_edges = triplets_to_dict(link_pred_splits['test'])
    valid_edges =triplets_to_dict( link_pred_splits['val'])
    featless_node_types=[]

    return train_edges, test_edges, valid_edges, g, featless_node_types

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

    g = pickle.load(open(os.path.join(data_folder, 'graph_reduced.pickle'), "rb")).to(device)

    # In[14]:
    labels = g.nodes['word'].data['features'][:, -1].cpu()


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
    category='word'
    num_classes=4
    featless_node_types = []
    if num_classes>1:
        labels_n = torch.zeros((np.shape(labels)[0], num_classes))
        if check_cuda:
            labels_n.cuda()
        for i in range(np.shape(labels)[0]):
            labels_n[i,int(labels[i])]=1
    else:
        labels_n=labels
    labels=labels_n
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

    # In[13]:

    G = pickle.load(open(os.path.join(data_folder, 'graph_0.001.pickle'), "rb")).to(device)

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
    category='history'
    num_classes=1
    featless_node_types = ['brand', 'customer', 'chain', 'market', 'dept', 'category', 'company']
    return train_idx,test_idx,val_idx,labels,G,category,num_classes,featless_node_types
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

    G = pickle.load(open(os.path.join(data_folder, 'graph_red.pickle'), "rb")).to(device)

    # extract adult label from graph
    label_type='genre'
    if label_type=='adult':
        labels = G.nodes['movie'].data['features'][:, 602]
        G.nodes['movie'].data['features'] = torch.cat(
        (G.nodes['movie'].data['features'][:, :602], G.nodes['movie'].data['features'][:, 603:]), 1)
    elif label_type=='genre':
        # last 30 are the genre labels
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
    num_classes=labels.shape[1]
    featless_node_types = []
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