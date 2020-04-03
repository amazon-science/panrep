from base_models import BaseRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import torch.nn as nn
from layers import RelGraphConvHetero, EmbeddingLayer,MiniBatchRelGraphConvHetero,MiniBatchRelGraphEmbed
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
    def __init__(self, in_dim,out_dim, etypes,ntype2id, num_hidden_layers=1,reg_param=0,use_cuda=False):
        super(DLinkPredictor, self).__init__()
        self.reg_param = reg_param
        self.etypes=etypes
        self.in_dim=in_dim
        self.ntype2id = ntype2id
        self.w_relation={}


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
            in_dim, out_dim, list(set(self.etypes)), "basis",
            num_bases=-1, activation=F.relu, self_loop=False,
            dropout=0))
        for i in range(self.num_hidden_layers-1):
            self.layers.append(RelGraphConvHetero(
                out_dim, out_dim, list(set(self.etypes)), "basis",
                num_bases=-1, activation=F.relu, self_loop=False,
                dropout=0))
    def calc_score(self,g, embedding, dict_s_d):
        # DistMult
        score={}

        for etype in self.etypes:
            (stype,e,dtype)=g.to_canonical_etype(etype)
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
    def forward(self, g,inp_h):
        h=inp_h
        for layer in self.layers:
            h = layer(g, h)
        return h
    def get_loss(self,g, embed, edict_s_d, e_dict_labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(g,embed, edict_s_d)
        predict_loss=0
        for etype in self.etypes:
            predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss(embed)

        return predict_loss + self.reg_param * reg_loss


class DLinkPredictorMB(nn.Module):
    def __init__(self,
                 g,
                 device,
                 h_dim,
                 num_rels,
                 num_bases=-1,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 regularization_coef=0,etype_key_map=None):
        super(DLinkPredictorMB, self).__init__()
        self.g = g
        self.device = device
        self.h_dim = h_dim
        self.etype_key_map=etype_key_map
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.regularization_coef = regularization_coef

        self.embed_layer = MiniBatchRelGraphEmbed(self.g, self.device, h_dim)
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation)

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(MiniBatchRelGraphConvHetero(
            self.h_dim, self.h_dim, self.rel_names, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            use_weight=False, dropout=self.dropout))
        # h2h
        for i in range(self.num_hidden_layers):
            self.layers.append(MiniBatchRelGraphConvHetero(
                self.h_dim, self.h_dim, self.rel_names, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        self.layers.to(self.device)

    def forward(self, p_blocks):
        p_h = self.embed_layer(p_blocks[0])

        for layer, block in zip(self.layers, p_blocks):
            p_h = layer(block, p_h)


        return p_h

    def regularization_loss(self, h_emb, t_emb, nh_emb, nt_emb):
        return torch.mean(h_emb.pow(2)) + \
               torch.mean(t_emb.pow(2)) + \
               torch.mean(nh_emb.pow(2)) + \
               torch.mean(nt_emb.pow(2)) + \
               torch.mean(self.w_relation.pow(2))

    def calc_pos_score(self, h_emb, t_emb, rids):
        # DistMult
        r = self.w_relation[rids]
        score = torch.sum(h_emb * r * t_emb, dim=-1)
        return score

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = heads.shape[1]
        r = self.w_relation[rids]
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = torch.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = tails.shape[1]
        r = self.w_relation[rids]
        heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
        heads = torch.transpose(heads, 1, 2)
        tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, heads)
    def get_loss(self,p_h, p_g, n_h,n_g, num_chunks, chunk_size, neg_sample_size):
        # loss calculation
        for ntype, emb in p_h.items():
            p_g.nodes[ntype].data['h'] = emb
        for ntype, emb in n_h.items():
            n_g.nodes[ntype].data['h'] = emb
        p_head_emb = []
        p_tail_emb = []
        rids = []
        for canonical_etype in p_g.canonical_etypes:
            head, tail = p_g.all_edges(etype=canonical_etype)
            head_emb = p_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = p_g.nodes[canonical_etype[2]].data['h'][tail]
            idx = self.etype_key_map[canonical_etype]
            rids.append(torch.full((head_emb.shape[0],), idx, dtype=torch.long))
            p_head_emb.append(head_emb)
            p_tail_emb.append(tail_emb)
        n_head_emb = []
        n_tail_emb = []
        for canonical_etype in n_g.canonical_etypes:
            head, tail = n_g.all_edges(etype=canonical_etype)
            head_emb = n_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = n_g.nodes[canonical_etype[2]].data['h'][tail]
            n_head_emb.append(head_emb)
            n_tail_emb.append(tail_emb)
        p_head_emb = torch.cat(p_head_emb, dim=0)
        p_tail_emb = torch.cat(p_tail_emb, dim=0)
        rids = torch.cat(rids, dim=0)
        n_head_emb = torch.cat(n_head_emb, dim=0)
        n_tail_emb = torch.cat(n_tail_emb, dim=0)
        assert p_head_emb.shape[0] == p_tail_emb.shape[0]
        assert rids.shape[0] == p_head_emb.shape[0]
        assert n_head_emb.shape[0] == n_tail_emb.shape[0]
        n_shuffle_seed = torch.randperm(n_head_emb.shape[0])
        n_head_emb = n_head_emb[n_shuffle_seed]
        n_tail_emb = n_tail_emb[n_shuffle_seed]

        loss = self.get_loss_h(p_head_emb,
                              p_tail_emb,
                              n_head_emb,
                              n_tail_emb,
                              rids,
                              num_chunks,
                              chunk_size,
                              neg_sample_size)
        return loss

    def get_loss_h(self, h_emb, t_emb, nh_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        pos_score = self.calc_pos_score(h_emb, t_emb, rids)
        t_neg_score = self.calc_neg_tail_score(h_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size)
        h_neg_score = self.calc_neg_head_score(nh_emb, t_emb, rids, num_chunks, chunk_size, neg_sample_size)
        pos_score = F.logsigmoid(pos_score)
        h_neg_score = h_neg_score.reshape(-1, neg_sample_size)
        t_neg_score = t_neg_score.reshape(-1, neg_sample_size)
        h_neg_score = F.logsigmoid(-h_neg_score).mean(dim=1)
        t_neg_score = F.logsigmoid(-t_neg_score).mean(dim=1)

        pos_score = pos_score.mean()
        h_neg_score = h_neg_score.mean()
        t_neg_score = t_neg_score.mean()
        predict_loss = -(2 * pos_score + h_neg_score + t_neg_score)

        reg_loss = self.regularization_loss(h_emb, t_emb, nh_emb, nt_emb)

        print("pos loss {}, neg loss {}|{}, reg_loss {}".format(pos_score.detach(),
                                                                h_neg_score.detach(),
                                                                t_neg_score.detach(),
                                                                self.regularization_coef * reg_loss.detach()))
        return predict_loss + self.regularization_coef * reg_loss


class End2EndClassifierRGCN(nn.Module):
    def __init__(self, h_dim, out_dim, num_rels,rel_names, num_bases,in_size_dict,ntypes,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False,h_dim_player=None):
        super(End2EndClassifierRGCN, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.rel_names = rel_names
        self.rel_names.sort()
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.in_size_dict = in_size_dict
        self.h_dim_player=h_dim_player

        self.embed_layer = EmbeddingLayer(self.in_size_dict, h_dim, ntypes)
        self.layers = nn.ModuleList()
        # h2h
        if h_dim_player is None:
            for i in range(self.num_hidden_layers):
                self.layers.append(RelGraphConvHetero(
                    self.h_dim, self.h_dim, self.rel_names, "basis",
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout))
        else:
            for i in range(self.num_hidden_layers):
                self.layers.append(RelGraphConvHetero(
                    self.h_dim_player[i], self.h_dim_player[i+1], self.rel_names, "basis",
                    self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                    dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvHetero(
            self.h_dim, self.out_dim, self.rel_names, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop))

    def forward(self,g):
        h = self.embed_layer(g)
        for layer in self.layers:
            h = layer(g, h)
        return h
    def forward_mb(self, h, blocks):
        h = self.embed_layer.forward_mb(h)
        for layer, block in zip(self.layers, blocks):
            h = layer.forward_mb(block, h)
        return h