from  classifiers import BaseRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn import RelGraphConv
from layers import RelGraphConvHetero, EmbeddingLayer, RelGraphAttentionHetero,MiniBatchRelGraphEmbed


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
class EncoderRelGraphConvHomo(nn.Module):
    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False,
                 layer_norm=False):
        super(EncoderRelGraphConvHomo, self).__init__()
        self.device = torch.device(device)
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm=layer_norm))
        # h2o
        #self.layers.append(RelGraphConv(
        #    self.h_dim, self.out_dim, self.num_rels, "basis",
        #    self.num_bases, activation=None,
        #    self_loop=self.use_self_loop,
        #    low_mem=self.low_mem, layer_norm=layer_norm))

    def forward(self, blocks, feats, corrupt=False, norm=None):
        h = feats
        if corrupt:
            perm = torch.randperm(len(feats))
            h = h[perm]
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h





class EncoderRelGraphConvHetero(nn.Module):
    def __init__(self,
                 h_dim,
                 etypes,
                 ntypes,
                 device,
                 g,
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

        self.embed_layer = MiniBatchRelGraphEmbed(g=g,device=device,embed_size=h_dim)
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
            for key in self.g.ntypes:
                # TODO possibly high complexity here??
                # The following implements the permutation of features within each node class.
                # for the negative sample in the information maximization step
                perm = torch.randperm(g.nodes[key].data['h_f'].shape[0])
                g.nodes[key].data['h_f'] = g.nodes[key].data['h_f'][perm]


        else:
            g = G

        h = self.embed_layer(g,full=True)

        for layer in self.layers:
            h = layer(g, h)
        return h
    def forward_mb(self,blocks,permute=False):

        h = self.embed_layer(blocks[0])

        if permute:
            for key in h.keys():
                perm = torch.randperm(h[key].shape[0])
                h[key] = h[key][perm]
        for layer, block in zip(self.layers, blocks):
            # print(h)
            h = layer.forward_mb(block, h)
        return h


class HGTLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 node_dict,
                 edge_dict,
                 n_heads,
                 dropout=0.2,
                 use_norm=False):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.total_rel = self.num_types * self.num_relations * self.num_types
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.att = None

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for t in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def edge_attention(self, edges):
        etype = edges.data['id'][0]

        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        relation_att = self.relation_att[etype]
        relation_pri = self.relation_pri[etype]
        key = torch.bmm(edges.src['k'].transpose(1, 0), relation_att).transpose(1, 0)
        att = (edges.dst['q'] * key).sum(dim=-1) * relation_pri / self.sqrt_dk

        '''
            Step 2: Heterogeneous Message Passing
        '''
        relation_msg = self.relation_msg[etype]
        val = torch.bmm(edges.src['v'].transpose(1, 0), relation_msg).transpose(1, 0)
        return {'a': att, 'v': val}

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        '''
            Softmax based on target node's id (edge_index_i).
            NOTE: Using DGL's API, there is a minor difference with this softmax with the original one.
                  This implementation will do softmax only on edges belong to the same relation type, instead of for all of the edges.
        '''
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G, h):
        with G.local_scope():
            node_dict, edge_dict = self.node_dict, self.edge_dict
            for srctype, etype, dsttype in G.canonical_etypes:
                k_linear = self.k_linears[node_dict[srctype]]
                v_linear = self.v_linears[node_dict[srctype]]
                q_linear = self.q_linears[node_dict[dsttype]]

                G.nodes[srctype].data['k'] = k_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                G.nodes[srctype].data['v'] = v_linear(h[srctype]).view(-1, self.n_heads, self.d_k)
                G.nodes[dsttype].data['q'] = q_linear(h[dsttype]).view(-1, self.n_heads, self.d_k)

                G.apply_edges(func=self.edge_attention, etype=etype)
            G.multi_update_all({etype: (self.message_func, self.reduce_func) \
                                for etype in edge_dict}, cross_reducer='mean')
            new_h = {}
            for ntype in G.ntypes:
                '''
                    Step 3: Target-specific Aggregation
                    x = norm( W[node_type] * gelu( Agg(x) ) + x )
                '''
                n_id = node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])
                trans_out = self.drop(self.a_linears[n_id](G.nodes[ntype].data['t']))
                trans_out = trans_out * alpha + h[ntype] * (1 - alpha)
                if self.use_norm:
                    new_h[ntype] = self.norms[n_id](trans_out)
                else:
                    new_h[ntype] = trans_out
            return new_h


class EncoderHGT(nn.Module):
    def __init__(self, G, node_dict, edge_dict, n_inp, n_hid, n_out, n_layers, n_heads, use_norm=True):
        super(EncoderHGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.gcs = nn.ModuleList()
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleList()
        for t in range(len(node_dict)):
            self.adapt_ws.append(nn.Linear(n_inp, n_hid))
        for _ in range(n_layers):
            self.gcs.append(HGTLayer(n_hid, n_hid, node_dict, edge_dict, n_heads, use_norm=use_norm))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, out_key):
        h = {}
        for ntype in G.ntypes:
            n_id = self.node_dict[ntype]
            h[ntype] = F.gelu(self.adapt_ws[n_id](G.nodes[ntype].data['inp']))
        for i in range(self.n_layers):
            h = self.gcs[i](G, h)
        return self.out(h[out_key])
