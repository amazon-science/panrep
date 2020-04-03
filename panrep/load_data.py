import os
import pickle
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import dgl
import numpy as np
import torch
from dgl.contrib.data import load_data
from sklearn.model_selection import StratifiedShuffleSplit,train_test_split

def load_hetero_data(args):
    if args.dataset == "kaggle_shoppers":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_kaggle_shoppers_data(args)
    elif args.dataset == "wn18":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_wn_data(args)
    elif args.dataset == "imdb":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_imdb_data(args)
    elif args.dataset == "imdb_preprocessed":
        train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types= load_imdb_preprocessed_data(args)
    else:
        raise NotImplementedError
    return train_idx,test_idx,val_idx,labels,g,category,num_classes,masked_node_types

def load_hetero_link_pred_data(args):
    if args.dataset == "wn":
        train_edges, test_edges, valid_edges, train_g, valid_g, test_g, featless_node_types = load_link_pred_wn_data(
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
    link_pred_splits=pickle.load(open(os.path.join(data_folder, 'link_pred_splits.pickle'), "rb"))#.to(torch.device("cpu"))
    #TODO finish the split of the edges check the saurav code
    #get eid from heterograph and use dgl.edge_subgraph
    train = 0.8
    test= 0.5
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
    test_g = dgl.heterograph(test_edges, num_nodes_per_types)
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

    data_folder = "../data/kg/wn18/"

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
    test_g = dgl.heterograph(test_edges, num_nodes_per_types)
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
    G = pickle.load(open(os.path.join(data_folder, 'graph.pickle'), "rb")).to(torch.device("cpu"))
    labels = pickle.load(open(os.path.join(data_folder, 'labels.pickle'), "rb"))
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
        if check_cuda:
            labels_n.cuda()
        for i in range(np.shape(labels)[0]):
            labels_n[i, int(labels[i])] = 1
    else:
        labels_n = labels
    labels = labels_n
    featless_node_types = []
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