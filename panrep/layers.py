import torch
from dgl import function as fn
from torch import nn as nn
from torch.nn import functional as F


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
            # TODO check that the masking step works
            g.nodes[srctype].data['h%d' % i] = torch.matmul(torch.matmul(
                g.nodes[srctype].data['x'], ws[etype]), g.edges[etype].data['mask'])
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        #  sum for the link prediction to not consider the zero messages
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


class EmbeddingLayer(nn.Module):
    def __init__(self, in_size_dict, out_size, ntypes):
        super(EmbeddingLayer, self).__init__()
        # W_r for each node
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size_dict[name], out_size, bias=False) for name in ntypes
        })

    def forward(self, G):
        for name in self.weight:
            G.apply_nodes(lambda nodes: {'h': self.weight[name](nodes.data['features'])}, ntype=name);
        hs = [G.nodes[ntype].data['h'] for ntype in G.ntypes]
        return hs


class RelGraphAttentionHetero(nn.Module):
    def __init__(self, in_feat, out_feat, etypes, bias=True, activation = None,
                 self_loop = False, dropout = 0.0):
        super(RelGraphAttentionHetero, self).__init__()
        # W_r for each relation
        self.attn_fc = nn.ModuleDict({etype: nn.Linear(2 * in_feat, 1, bias=False) for etype in etypes})
        self.message_fc = {etype: self.message_func_etype(etype) for etype in etypes}
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

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

    def message_func_etype(self, etype):
        def message_func(edges):
            e = torch.cat([edges.src['Wh_%s' % etype], edges.dst['h']], dim=1);
            e = self.attn_fc[etype](e);
            return {'m': edges.src['Wh_%s' % etype], 'e': e};

        return message_func;

    def edge_attention(self, edges):

        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        num_edges = nodes.mailbox['e'].size(1);
        check_cuda=False
        if next(self.parameters()).is_cuda:
            check_cuda = True

        device = torch.device("cuda:" + str(torch.cuda.current_device()) if check_cuda else "cpu")
        mask = torch.empty(nodes.mailbox['e'].size(),  device=device,dtype=torch.uint8).random_(0, num_edges) >= 201
        temp = torch.where(mask, torch.empty(nodes.mailbox['e'].size(),  device=device, dtype=torch.float).fill_(
            float('-inf')), nodes.mailbox['e']);
        alpha = F.softmax(temp, dim=1)
        # equation (4)
        x = torch.sum(alpha * nodes.mailbox['m'], dim=1)
        return {'x': x}

    def forward(self, g, xs):
        g = g.local_var()
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['xs'] = xs[i]
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in g.canonical_etypes:
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            g.nodes[srctype].data['Wh_%s' % etype] = g.nodes[srctype].data['xs'];
            funcs[etype] = (self.message_fc[etype], self.reduce_func)
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        # TODO sum for the link prediction to not consider the zero messages
        g.multi_update_all(funcs, 'sum')

        hs = [g.nodes[ntype].data['x'] for ntype in g.ntypes]


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
        #Update for feature use
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['h'] = hs[i]
        return hs