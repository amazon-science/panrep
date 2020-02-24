from base_models import BaseRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import dgl.function as fn
import torch.nn.functional as F
import torch.nn as nn

#TODO SEE DGI code architecture and inspire similar context. add the dgi losses

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


class RelGraphConvHetero(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : int
        Relation names.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvHetero, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_rels = len(rel_names)
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
        else:
            raise ValueError("Only basis regularizer is supported.")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_weight(self):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        return {self.rel_names[i]: w.squeeze(0) for i, w in enumerate(torch.split(weight, 1, dim=0))}

    def forward(self, g, xs):
        """ Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        xs : list of torch.Tensor
            Node feature for each node type.

        Returns
        -------
        list of torch.Tensor
            New node features for each node type.
        """
        g = g.local_var()
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = xs[i]
        ws = self.basis_weight()
        funcs = {}
        for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
            g.nodes[srctype].data['h%d' % i] = torch.matmul(
                g.nodes[srctype].data['x'], ws[etype])
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]
        for i in range(len(hs)):
            h = hs[i]
            # apply bias and activation
            if self.self_loop:
                h = h + torch.matmul(xs[i], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            hs[i] = h
        return hs


class HeteroRGCNLayerFirst(nn.Module):
    def __init__(self, in_size_dict, out_size, ntypes):
        super(HeteroRGCNLayerFirst, self).__init__()
        # W_r for each node
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size_dict[name], out_size, bias=False) for name in ntypes
        })

    def forward(self, G):
        for name in self.weight:
            G.apply_nodes(lambda nodes: {'h': self.weight[name](nodes.data['features'])}, ntype=name);
        hs = [G.nodes[ntype].data['h'] for ntype in G.ntypes]
        return hs


class EncoderRelGraphConvHetero(nn.Module):
    def __init__(self,
                 G,
                 h_dim, out_dim,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False):
        super(EncoderRelGraphConvHetero, self).__init__()
        self.G = G
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(G.etypes))
        self.rel_names.sort()
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        self.use_self_loop = use_self_loop
        self.in_size_dict = {};
        for name in self.G.ntypes:
            self.in_size_dict[name] = self.G.nodes[name].data['features'].size(1);

        self.embed_layer = HeteroRGCNLayerFirst(self.in_size_dict, h_dim, G.ntypes)
        self.layers = nn.ModuleList()
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))

    def forward(self, corrupt=False):
        if corrupt:
            # create local variable do not permute the original graph
            g = self.G.local_var()
            for key in self.in_size_dict:
                # TODO possibly high complexity here??
                # The following implements the permutation of features within each node class.
                # for the negative sample in the information maximization step
                perm = torch.randperm(g.nodes[key].data['features'].shape[0])
                g.nodes[key].data['features'] = g.nodes[key].data['features'][perm]


        else:
            g = self.G

        h = self.embed_layer(g)
        for layer in self.layers:
            h = layer(g, h)
        return h
