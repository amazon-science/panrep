'''
This file contains the definition for different layers used in the GNN models.

'''

import torch
from dgl import function as fn
from torch import nn as nn
from torch.nn import functional as F
import dgl
from torch.cuda import nvtx

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
        #if self.self_loop:
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
        #g = g.local_var()
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = xs[ntype]
        ws = self.basis_weight()
        funcs = {}
        for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
            # TODO check that the masking step works
            g.nodes[srctype].data['h%d' % i] = torch.matmul(
                g.nodes[srctype].data['x'], ws[etype])# g.edges[etype].data['mask'])
            # TODO use sum instead of mean
            funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
            # TODO check the masked 1 with without mask that returns the same
            #funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = {}
        for ntype in g.ntypes:
            if 'h' in g.nodes[ntype].data:
                hs[ntype] = g.nodes[ntype].data['h']
            else:
                hs[ntype]=torch.matmul(xs[ntype][:g.number_of_nodes(ntype)], self.loop_weight)

        def _apply(h,ntype):
            # apply bias and activation
            if self.self_loop:
                h = h + torch.matmul(xs[ntype][:h.shape[0]], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h

        hs = {ntype: _apply(h,ntype) for ntype, h in hs.items()}
        return hs


    def forward_mb(self, g, xs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input block graph.
        xs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        list of torch.Tensor
            New node features for each node type.
        """
        #g = g.local_var()
        for ntype, x in xs.items():
            if ntype in g.srctypes:
                g.srcnodes[ntype].data['x'] = x

        ws = self.basis_weight()
        funcs = {}
        for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
                if srctype not in xs:
                    continue
                g.srcnodes[srctype].data['h%d' % i] = torch.matmul(
                    g.srcnodes[srctype].data['x'], ws[etype])
                funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = {}
        for ntype in g.dsttypes:
            if 'h' in g.dstnodes[ntype].data:
                hs[ntype] = g.dstnodes[ntype].data['h']
            else:
                hs[ntype]=torch.matmul(xs[ntype][:g.number_of_dst_nodes(ntype)], self.loop_weight)
        def _apply(h,ntype):
            # apply bias and activation
            if self.self_loop:
                h = h + torch.matmul(xs[ntype][:h.shape[0]], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
        hs = {ntype : _apply(h,ntype) for ntype, h in hs.items()}
        return hs

class RelGraphConvHomo(nn.Module):
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 low_mem=False,
                 dropout=0.0,
                 layer_norm=False):
        super(RelGraphConvHomo, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        assert low_mem
        assert regularizer == "basis"

        # cached parameters for low mem version
        self._etypes = None

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
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % self.num_bases != 0 or out_feat % self.num_bases != 0:
                raise ValueError(
                    'Feature size must be a multiplier of num_bases (%d).'
                    % self.num_bases
                )
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(torch.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        nvtx.range_push("generate_weight")
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        nvtx.range_pop()

        # calculate msg @ W_r before put msg into edge
        # if src is torch.int64 we expect it is an index select
        device = edges.src['h'].device
        nvtx.range_push("low_mem_forward")

        nvtx.range_push("split")
        h = torch.split(edges.src['h'], self.section)
        nvtx.range_pop()

        msg = []
        for etype in range(self.num_rels):
            if h[etype].shape[0] == 0:
                continue
            nvtx.range_push("select_weight")
            w = weight[etype]
            nvtx.range_pop()

            nvtx.range_push("matmul_src_w")
            sub_msg = torch.matmul(h[etype], w)
            nvtx.range_pop()

            msg.append(sub_msg)

        nvtx.range_push("concat")
        msg = torch.cat(msg)
        nvtx.range_pop()

        nvtx.range_pop()  # layer forward

        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, feat, etypes, norm, section):
        with g.local_scope():
            g.srcdata['h'] = feat
            g.edata['type'] = etypes
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = matmul_maybe_select(feat[:g.number_of_dst_nodes()], self.loop_weight)
            # message passing
            self.section = section
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
            self.section = None
            # apply bias and activation
            node_repr = g.dstdata['h']
            if self.layer_norm:
                node_repr = self.layer_norm_weight(node_repr)
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr
def matmul_maybe_select(A, B):
    if A.dtype == torch.int64 and len(A.shape) == 1:
        return B.index_select(0, A)
    else:
        return torch.matmul(A, B)
class RelGraphEmbedLayerHomo(nn.Module):
    r"""Embedding layer for featureless heterograph.
    Parameters
    ----------
    dev_id : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    node_tides : tensor
        Storing the node type id for each node starting from 0
    num_of_ntype : int
        Number of node types
    input_size : list of int
        A list of input feature size for each node type. If None, we then
        treat certain input feature as an one-hot encoding feature.
    embed_size : int
        Output embed size
    embed_name : str, optional
        Embed name
    """
    def __init__(self,
                 dev_id,
                 num_nodes,
                 node_tids,
                 num_of_ntype,
                 input_size,
                 embed_size,
                 sparse_emb=False,
                 embed_name='embed'):
        super(RelGraphEmbedLayerHomo, self).__init__()
        self.dev_id = dev_id
        self.embed_size = embed_size
        self.embed_name = embed_name
        self.num_nodes = num_nodes
        self.sparse_emb = sparse_emb

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        self.num_of_ntype = num_of_ntype
        self.idmap = torch.empty(num_nodes).long()

        for ntype in range(num_of_ntype):
            if input_size[ntype] is not None:
                input_emb_size = input_size[ntype].shape[1]
                embed = nn.Parameter(torch.Tensor(input_emb_size, self.embed_size))
                nn.init.xavier_uniform_(embed)
                self.embeds[str(ntype)] = embed

        self.node_embeds = torch.nn.Embedding(node_tids.shape[0], self.embed_size, sparse=self.sparse_emb)
        nn.init.uniform_(self.node_embeds.weight, -1.0, 1.0)

    def forward(self, node_ids, node_tids, type_ids, features):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_tids : tensor
            node type ids
        features : list of features
            list of initial features for nodes belong to different node type.
            If None, the corresponding features is an one-hot encoding feature,
            else use the features directly as input feature and matmul a
            projection matrix.
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        tsd_ids = node_ids.to(self.node_embeds.weight.device)
        idx = torch.empty(node_ids.shape[0], dtype=torch.int64, device=self.dev_id)
        embeds = []
        num_nodes = 0
        for ntype in range(self.num_of_ntype):
            if features[ntype] is not None:
                loc = node_tids == ntype
                embeds.append(features[ntype][type_ids[loc]].to(self.dev_id) @ self.embeds[str(ntype)].to(self.dev_id))
            else:
                loc = node_tids == ntype
                embeds.append(self.node_embeds(tsd_ids[loc]).to(self.dev_id))
            idx[loc] = torch.arange(len(embeds[-1]), device=self.dev_id) + num_nodes
            num_nodes += len(embeds[-1])
        embeds = torch.cat(embeds)
        return embeds[idx]
class EmbeddingLayer(nn.Module):
    def __init__(self, in_size_dict, out_size, ntypes):
        super(EmbeddingLayer, self).__init__()
        # W_r for each node
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size_dict[name], out_size, bias=False) for name in ntypes
        })

    def forward(self, G):
        hs={}
        for name in self.weight:
            G.apply_nodes(lambda nodes: {'h': self.weight[name](nodes.data['h_f'])}, ntype=name);
        for ntype in self.weight:
            hs[ntype]=G.nodes[ntype].data['h']
        return hs
    def forward_mb(self, embeddings):
        # TODO implement this layer
        hs={}
        for ntype in self.weight:
            hs[ntype]=self.weight[ntype](embeddings[ntype])
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


class MiniBatchRelGraphConvHetero(nn.Module):
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
    use_weight : bool, optional
        If True, multiply the input node feature with a learnable weight matrix
        before message passing.
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
                 use_weight=True,
                 dropout=0.0):
        super(MiniBatchRelGraphConvHetero, self).__init__()
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

        self.use_weight = use_weight
        if use_weight:
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
        return {self.rel_names[i] : w.squeeze(0) for i, w in enumerate(torch.split(weight, 1, dim=0))}

    def forward(self, g, xs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input block graph.
        xs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        list of torch.Tensor
            New node features for each node type.
        """
        g = g.local_var()
        for ntype, x in xs.items():
            g.srcnodes[ntype].data['x'] = x
        if self.use_weight:
            ws = self.basis_weight()
            funcs = {}
            for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
                if srctype not in xs:
                    continue
                g.srcnodes[srctype].data['h%d' % i] = torch.matmul(
                    g.srcnodes[srctype].data['x'], ws[etype])
                funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        else:
            funcs = {}
            for i, (srctype, etype, dsttype) in enumerate(g.canonical_etypes):
                if srctype not in xs:
                    continue
                g.srcnodes[srctype].data['h%d' % i] = g.srcnodes[srctype].data['x']
                funcs[(srctype, etype, dsttype)] = (fn.copy_u('h%d' % i, 'm'), fn.mean('m', 'h'))
        # message passing
        g.multi_update_all(funcs, 'sum')

        hs = {}
        for ntype in g.dsttypes:
            if 'h' in g.dstnodes[ntype].data:
                hs[ntype] = g.dstnodes[ntype].data.pop('h')
        def _apply(ntype, h):
            # apply bias and activation
            if self.self_loop:
                h = h + torch.matmul(xs[ntype][:h.shape[0]], self.loop_weight)
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
        hs = {ntype : _apply(ntype, h) for ntype, h in hs.items()}
        return hs

class MiniBatchRelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""
    def __init__(self,
                 g,
                 device,
                 embed_size):
        super(MiniBatchRelGraphEmbed, self).__init__()
        self.g = g
        self.device = device
        self.embed_size = embed_size

        # create weight embeddings for each node for each relation
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                embed = nn.Parameter(torch.Tensor(g.nodes[ntype].data["h_f"].shape[1], self.embed_size).to(self.device))
                nn.init.xavier_uniform_(embed)
            else:
                embed = nn.Parameter(torch.Tensor(g.number_of_nodes(ntype), self.embed_size))
                nn.init.xavier_uniform_(embed)
            self.embeds[ntype] = embed

    def forward(self, g,full=False):
        """Forward computation

        Parameters
        ----------
        block : DGLHeteroGraph, optional
            If not specified, directly return the full graph with embeddings stored in
            :attr:`embed_name`. Otherwise, extract and store the embeddings to the block
            graph and return.

        Returns
        -------
        DGLHeteroGraph
            The block graph fed with embeddings.
        """
        emb = {}
        for ntype in g.srctypes:
            if g.srcnodes[ntype].data.get('h_f', None) is not None:
                emb[ntype] = g.srcnodes[ntype].data['h_f'] @ self.embeds[ntype]
            else:
                if not full:
                    emb[ntype] = self.embeds[ntype][g.srcnodes[ntype].data[dgl.NID]]#.to(self.device)
                else:
                    emb[ntype] = self.embeds[ntype][torch.tensor(range(g.number_of_nodes(ntype)))]#.to("cpu"))
        return emb
