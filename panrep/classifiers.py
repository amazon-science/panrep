from base_models import BaseRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import torch.nn as nn
from layers import RelGraphConvHetero, EmbeddingLayer
import torch.nn.functional as F


class ClassifierMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size,out_size):
        super(ClassifierMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size=out_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.out_size)
        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        #output = self.sigmoid(output)
        return output
class ClassifierRGCN(BaseRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RelGraphConv(self.inp_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)
    # TODO different layers may have different number of hidden units current implementation prevents
    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)
    def build_class_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, dim=1),
                self_loop=self.use_self_loop)

    def build_reconstruct_output_layer(self):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, dim=1),
                self_loop=self.use_self_loop)
    def build_output_layer(self):
        return self.build_class_output_layer()

class DLinkPredictor(nn.Module):
    def __init__(self, in_dim,out_dim, G, num_hidden_layers=1,reg_param=0,use_cuda=False):
        super(DLinkPredictor, self).__init__()
        self.reg_param = reg_param
        self.etypes=G.etypes
        self.G=G
        self.in_dim=in_dim
        self.w_relation={}
        self.ntype2id={}
        for i, ntype in enumerate(self.G.ntypes):
            self.ntype2id[ntype] = i
        for ename in self.etypes:
            if use_cuda:
                self.w_relation[ename] = nn.Parameter(torch.Tensor(out_dim,1)).cuda()
            else:
                self.w_relation[ename] = nn.Parameter(torch.Tensor(out_dim,1))
            nn.init.xavier_uniform_(self.w_relation[ename],
                                gain=nn.init.calculate_gain('relu'))
        # one hidden layer
        self.num_hidden_layers=num_hidden_layers
        self.layers= nn.ModuleList()
        self.layers.append(RelGraphConvHetero(
            in_dim, out_dim, list(set(self.G.etypes)), "basis",
            num_bases=-1, activation=F.relu, self_loop=False,
            dropout=0))
        for i in range(self.num_hidden_layers-1):
            self.layers.append(RelGraphConvHetero(
                out_dim, out_dim, list(set(self.G.etypes)), "basis",
                num_bases=-1, activation=F.relu, self_loop=False,
                dropout=0))
    def calc_score(self, embedding, dict_s_d):
        # DistMult
        score={}
        for etype in self.etypes:
            (stype,e,dtype)=self.G.to_canonical_etype(etype)
            s = embedding[self.ntype2id[stype]][dict_s_d[etype][:, 0]]
            r = self.w_relation[etype].squeeze()
            o = embedding[self.ntype2id[dtype]][dict_s_d[etype][:, 1]]
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
            loss=0
            for e in embedding:
                loss+=torch.mean(e.pow(2))

            for e in self.w_relation.keys():
                loss+=torch.mean(self.w_relation[e].pow(2))
            return loss
    def forward(self, inp_h):
        h=inp_h
        for layer in self.layers:
            h = layer(self.G, h)
        return h
    def get_loss(self, embed, edict_s_d, e_dict_labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, edict_s_d)
        predict_loss=0
        for etype in self.etypes:
            predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss(embed)

        return predict_loss + self.reg_param * reg_loss
class End2EndClassifierRGCN(nn.Module):
    def __init__(self, G, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(End2EndClassifierRGCN, self).__init__()
        self.G=G
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rel_names = list(set(G.etypes))
        self.rel_names.sort()
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.in_size_dict = {}
        for name in self.G.ntypes:
            self.in_size_dict[name] = self.G.nodes[name].data['features'].size(1);
        self.embed_layer = EmbeddingLayer(self.in_size_dict, h_dim, G.ntypes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.out_dim, self.rel_names, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self):
        h = self.embed_layer(self.G)
        for layer in self.layers:
            h = layer(self.G, h)
        return h