from base_models import BaseRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import RelGraphConvHetero, EmbeddingLayer, RelGraphAttentionHetero


class EncoderRGCN(BaseRGCN):
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
        return self.build_reconstruct_output_layer()


class EncoderRelGraphAttentionHetero(nn.Module):
    def __init__(self,
                 h_dim,
                 in_size_dict,
                 etypes,
                 ntypes,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EncoderRelGraphAttentionHetero, self).__init__()
        self.h_dim = h_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.use_self_loop = use_self_loop
        self.in_size_dict = in_size_dict


        self.embed_layer = EmbeddingLayer(self.in_size_dict, h_dim, ntypes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphAttentionHetero(
                self.h_dim, self.h_dim, etypes, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))

    def forward(self,G, corrupt=False):
        if corrupt:
            # create local variable do not permute the original graph
            g = G.local_var()
            for key in self.in_size_dict:
                # TODO possibly high complexity here??
                # The following implements the permutation of features within each node class.
                # for the negative sample in the information maximization step
                perm = torch.randperm(g.nodes[key].data['features'].shape[0])
                g.nodes[key].data['features'] = g.nodes[key].data['features'][perm]


        else:
            g = G

        h = self.embed_layer(g)
        for layer in self.layers:
            h = layer(g, h)
        return h


class EncoderRelGraphConvHetero(nn.Module):
    def __init__(self,
                 h_dim,
                 in_size_dict,
                 etypes,
                 ntypes,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EncoderRelGraphConvHetero, self).__init__()
        self.h_dim = h_dim
        self.rel_names = list(set(etypes))
        self.rel_names.sort()
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.use_self_loop = use_self_loop
        self.in_size_dict = in_size_dict

        self.embed_layer = EmbeddingLayer(self.in_size_dict, h_dim, ntypes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))

    def forward(self,G, corrupt=False):
        if corrupt:
            # create local variable do not permute the original graph
            g = G.local_var()
            for key in self.in_size_dict:
                # TODO possibly high complexity here??
                # The following implements the permutation of features within each node class.
                # for the negative sample in the information maximization step
                perm = torch.randperm(g.nodes[key].data['features'].shape[0])
                g.nodes[key].data['features'] = g.nodes[key].data['features'][perm]


        else:
            g = G

        h = self.embed_layer(g)
        for layer in self.layers:
            h = layer(g, h)
        return h
    def forward_mb(self, embedding,blocks):

        h = self.embed_layer.forward_mb(embedding)
        for layer, block in zip(self.layers, blocks):
            h = layer.forward_mb(block, h)
        return h
