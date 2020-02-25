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