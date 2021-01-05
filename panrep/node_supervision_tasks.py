'''
This file contains the implementation of the decoders used to supervise PanRep.
'''

import torch.nn.functional as F
import torch.nn as nn
import torch
import math
from layers import RelGraphConvHetero
from functools import partial
from sklearn.metrics import roc_auc_score
import dgl
from torch.autograd import Variable as V

class MetapathRWalkerSupervision(nn.Module):
        def __init__(self, in_dim, out_dim=0,  num_hidden_layers=0, reg_param=0,negative_rate=1,device=None,mrw_interact=None):
            super(MetapathRWalkerSupervision, self).__init__()
            self.reg_param = reg_param
            self.in_dim = in_dim
            self.device=device
            self.negative_rate=negative_rate
            self.num_hidden_layers = num_hidden_layers
            if self.num_hidden_layers==0:
                out_dim=in_dim
            self.mrw_interact=mrw_interact
            w_relation={}
            self.w_relation=nn.ParameterDict()
            for ntype in mrw_interact.keys():
                for neighbor_ntype in mrw_interact[ntype]:
                    ename=ntype+neighbor_ntype
                    w_relation[ename] = nn.Parameter(torch.Tensor(out_dim, 1))
                    nn.init.xavier_uniform_(w_relation[ename],
                                            gain=nn.init.calculate_gain('relu'))
            self.w_relation.update(w_relation)
        def calc_score(self, head_embed, tail_embed,etype):
                # DistMult
                s = head_embed
                r = self.w_relation[etype].squeeze()
                # TODO consider other formulations metapath2vec
                o = tail_embed
                score = torch.sum(s * r * o, dim=1)
                return score
        # TODO 2 different type of embeddings context and embedding to improve performance
        #  Downsample frequent nodes..
        #   Suppose there is a high degree node.
        #   Popular nodes deal by inverse scaling with the degree.
        #    Define rw that probability parameter inverse degree.. downsample
        def regularization_loss(self, embedding):
            loss = 0
            for ntype in embedding.keys():
                loss += torch.mean(embedding[ntype].pow(2))

            for e in self.w_relation.keys():
                loss += torch.mean(self.w_relation[e].pow(2))
            return loss

        def forward(self, g, inp_h):
            return inp_h

        def get_loss(self, g, embed,rw_neighbors):

            predict_loss = 0
            for ntype in rw_neighbors.keys():
                cur_ntype_neighbors=rw_neighbors[ntype]
                for neighbo_ntype in cur_ntype_neighbors.keys():
                        neighbor_ids = cur_ntype_neighbors[neighbo_ntype][g.dstnodes[ntype].data['_ID']]
                        # Build inverse mapping given the ids in the original graph it returns the ids in the seed graph.
                        # This is given by the inverse of the seed nodes....
                        sampled_neighbors_ids = g.dstnodes[neighbo_ntype].data['_ID']

                        neighbors_ids_in_sampled_g = (torch.nonzero(
                            neighbor_ids[..., None] == sampled_neighbors_ids))
                        if neighbors_ids_in_sampled_g.shape[0]>0:
                            head_id = neighbors_ids_in_sampled_g[:, 0]  # head id
                            column_neighbor_ids_ind = neighbors_ids_in_sampled_g[:, 1]
                            tail_ids = neighbors_ids_in_sampled_g[:, 2]  # tail id?
                            tail_embedding = embed[neighbo_ntype][tail_ids]
                            head_embedding = embed[ntype][head_id]
                            etype=ntype+neighbo_ntype
                            pos_cur_score = self.calc_score(head_embedding, tail_embedding,etype)
                            predict_loss += F.binary_cross_entropy_with_logits(pos_cur_score,
                                                                               torch.ones((tail_embedding.shape[0]), device=self.device))
                            # negative pairs
                            # perturb tail
                            neg_tail_id = torch.randint(embed[neighbo_ntype].shape[0], (tail_embedding.shape[0]*self.negative_rate,))
                            neg_head_id=head_id.repeat(self.negative_rate)
                            neg_cur_score= self.calc_score(embed[ntype][neg_head_id],embed[neighbo_ntype][neg_tail_id],etype)
                            predict_loss += F.binary_cross_entropy_with_logits(neg_cur_score,
                                                                               torch.zeros((neg_head_id.shape[0]), device=self.device))
                        # load results from finetune_panrep_node_classification_mb


            reg_loss = self.regularization_loss(embed)
            if self.reg_param == 0:
                return predict_loss
            return predict_loss + self.reg_param * reg_loss


class LinkPredictorDistMultMB(nn.Module):
    def __init__(self,
                 g,
                 device,
                 h_dim,
                 num_rels,
                 dropout=0,
                 use_self_loop=False,
                 regularization_coef=0,etype_key_map=None):
        super(LinkPredictorDistMultMB, self).__init__()
        self.g = g
        self.device = device
        self.h_dim = h_dim
        self.etype_key_map=etype_key_map
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.regularization_coef = regularization_coef

        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation)



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
class LinkPredictorLearnableEmbed(nn.Module):
    def __init__(self, out_dim, etypes, ntype2id,reg_param=0,use_cuda=False,edg_pct=0.8,ng_rate=5,activation=nn.ReLU(),filtered=False):
        super(LinkPredictorLearnableEmbed, self).__init__()
        self.reg_param = reg_param
        self.etypes=etypes
        self.w_relation=nn.ParameterDict()
        self.ng_rate=ng_rate
        self.filtered = filtered
        self.ntype2id=ntype2id
        self.edg_pct=edg_pct
        self.use_cuda=use_cuda
        self.out_dim=out_dim
        # TODO Parameterize per node type pairs!!
        layers=[]
        layers.append(nn.Linear(2*self.out_dim, 2*self.out_dim))
        layers.append(activation)
        layers.append(nn.Linear(2*self.out_dim, self.out_dim, bias=True))
        self.weight = nn.Sequential(*layers)

    def calc_pos_score_with_rids_per_rel(self,h_emb, t_emb, rids,etypes2ids=None,device=None):
        # DistMult
        rels={}
        for i in range(rids.shape[0]):
            if rids[i].item() not in rels:
                rels[rids[i].item()]=[i]
            else:
                rels[rids[i].item()] += [i]
        score=torch.zeros((len(rids))).to(device)
        for key in rels.keys():
            s = h_emb[rels[key]]

            o = t_emb[rels[key]]

            r = self.weight(torch.cat((s,o),axis=1))#.mean(axis=0).repeat((o.shape[0],1))
            score[rels[key]] = torch.sum(s * r * o, dim=1)
        return score


    def calc_pos_score_with_rids(self,h_emb, t_emb, rids,etypes2ids=None,device=None):
        # DistMult
        rels={}
        for i in range(rids.shape[0]):
            if rids[i].item() not in rels:
                rels[rids[i].item()]=[i]
            else:
                rels[rids[i].item()] += [i]
        score=torch.zeros((len(rids))).to(device)
        for key in rels.keys():
            s = h_emb[rels[key]]

            o = t_emb[rels[key]]

            r = self.weight(torch.cat((s,o),axis=1)).mean(axis=0).repeat((o.shape[0],1))
            score[rels[key]] = torch.sum(s * r * o, dim=1)
        return score

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size,etypes2ids, device=None):
        hidden_dim = heads.shape[1]
        exp_tails = tails.repeat((int(heads.shape[0] / tails.shape[0]), 1))
        rem=heads.shape[0]-tails.shape[0]*int(heads.shape[0] / tails.shape[0])

        if rem>0:
            exp_tails=torch.cat( (exp_tails,tails[:rem]), dim=0)

        r = self.weight(torch.cat((heads, exp_tails), axis=1)).mean(axis=0).repeat((heads.shape[0],1))
        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = torch.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size,etypes2ids, device=None):
        hidden_dim = tails.shape[1]
        exp_heads = heads.repeat((int(tails.shape[0] / heads.shape[0]), 1))
        rem=tails.shape[0]-heads.shape[0]*int(tails.shape[0] / heads.shape[0])

        if rem > 0:
            exp_heads = torch.cat((exp_heads, heads[:rem]), dim=0)

        r = self.weight(torch.cat((exp_heads, tails), axis=1)).mean(axis=0).repeat((tails.shape[0],1))
        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
        heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
        heads = torch.transpose(heads, 1, 2)
        tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, heads)

    def calc_score(self, g,embedding, dict_s_d):
        # DistMult
        score={}

        for etype in self.etypes:
            (stype,e,dtype)=g.to_canonical_etype(etype)
            s = embedding[self.ntype2id[stype]][dict_s_d[etype][:, 0]]
            r = self.w_relation[etype].squeeze()
            o = embedding[self.ntype2id[dtype]][dict_s_d[etype][:, 1]]
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def calc_score_mb(self, g,embedding, dict_s_d):
        # DistMult
        score = {}
        for etype in dict_s_d.keys():
            (stype, e, dtype)=g.to_canonical_etype(etype)
            s = embedding[stype][dict_s_d[etype][0]]

            o = embedding[dtype][dict_s_d[etype][1]]

            r = self.weight(torch.cat((s,o),axis=1)).mean(axis=0).repeat((o.shape[0],1))
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
            loss=0
            for e in embedding:
                loss+=torch.mean(e.pow(2))

            for e in self.w_relation.keys():
                loss+=torch.mean(self.w_relation[e].pow(2))
            return loss

    def forward(self, g,embed, edict_s_d, e_dict_labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(g,embed, edict_s_d)
        predict_loss=0
        for etype in self.etypes:
            predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss(embed)

        return predict_loss + self.reg_param * reg_loss

    def generate_samples(self,g):
        # TODO move to dataloader so that it is faster
        edg_pct=self.edg_pct
        ng_rate=self.ng_rate
        filtered=self.filtered
        e_dict_labels={}
        edict_s_d={}
        for etype in g.etypes:
            u, v = g.all_edges(etype=etype)
            (srctype, ety, desttype) = g.to_canonical_etype(etype)
            # keep only pairs of nodes between dstnodes
            src_len = g.dstnodes[srctype].data['_ID'].shape[0]
            dest_len = g.dstnodes[desttype].data['_ID'].shape[0]
            src_nodes_in_dest = u < src_len
            if src_len==0 or dest_len==0 or torch.sum(src_nodes_in_dest)==0:
                continue
            else:
                pos_head = u[src_nodes_in_dest]
                pos_tail = v[src_nodes_in_dest]
                # filter for edg_pct
                pos_head = pos_head[:int(edg_pct * pos_head.shape[0])]
                pos_tail = pos_tail[:int(edg_pct * pos_tail.shape[0])]
                size_of_batch = pos_head.shape[0]
                labels = torch.zeros(size_of_batch * (ng_rate + 1)).float()
                if self.use_cuda:
                    labels=labels.cuda()
                labels[: size_of_batch] = 1
                head = pos_head.repeat((ng_rate + 1))
                tail = pos_tail.repeat((ng_rate + 1))
                # TODO filter edges
                #  set_pos_edges ={(1,3),(1,2)}
                #  set_pos_edges = {(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2)}
                #  neg_edges=set_neg_edges.minus(set_pos_edges)
                head[size_of_batch:] = torch.randint(src_len, (size_of_batch * ng_rate,))
                if filtered:
                    pos_pair = set(tuple(map(tuple, torch.cat((pos_head.unsqueeze(0), pos_tail.unsqueeze(0)), dim=0).transpose(1,0).cpu().numpy())))
                    neg_pair = set(tuple(map(tuple,torch.cat((head[size_of_batch:].unsqueeze(0),
                                          tail[size_of_batch:].unsqueeze(0)), dim=0).transpose(1,0).cpu().numpy())))
                    filt_neg_pair=neg_pair.difference(pos_pair)
                e_dict_labels[etype]=labels
                edict_s_d[etype]=(head,tail)
        return e_dict_labels,edict_s_d
    def regularization_loss_mb(self, embedding):
            loss = 0
            for ntype in embedding.keys():
                loss += torch.mean(embedding[ntype].pow(2))

            for e in self.w_relation.keys():
                loss += torch.mean(self.w_relation[e].pow(2))
            return loss
    def forward_mb(self, g,embed):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)

        e_dict_labels,edict_s_d=self.generate_samples(g=g)
        score = self.calc_score_mb(g,embed, edict_s_d)
        predict_loss=0
        for etype in e_dict_labels.keys():
            if len(score[etype])>0:
                predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss_mb(embed)

        return predict_loss + self.reg_param * reg_loss
class LinkPredictorHomo(nn.Module):
    def __init__(self, h_dim, num_rels,ng_rate,use_cuda, reg_param=0):
        super(LinkPredictorHomo, self).__init__()
        self.reg_param = reg_param
        self.ng_rate=ng_rate
        self.use_cuda=use_cuda
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score
    def generate_samples(self,g):
        # TODO move to dataloader so that it is faster
        ng_rate=self.ng_rate

        etypes = g.edata[dgl.ETYPE]
        u, v = g.all_edges()
        src_len = g.dstnodes['_N'].data[dgl.NID].shape[0]
        dest_len = g.dstnodes['_N'].data[dgl.NID].shape[0]
        src_nodes_in_dest = u < src_len
        pos_head = u[src_nodes_in_dest]
        pos_tail = v[src_nodes_in_dest]
        pos_etypes=etypes[src_nodes_in_dest]
        if src_len == 0 or dest_len == 0 or torch.sum(src_nodes_in_dest) == 0:
            return None,None


        size_of_batch = pos_head.shape[0]
        labels = torch.zeros(size_of_batch * (ng_rate + 1)).float()
        if self.use_cuda:
            labels=labels.cuda()
        labels[: size_of_batch] = 1
        head = pos_head.repeat((ng_rate + 1))
        tail = pos_tail.repeat((ng_rate + 1))
        etypes=pos_etypes.repeat((ng_rate + 1))
        head[size_of_batch:] = torch.randint(src_len, (size_of_batch * ng_rate,))
        triplets=torch.cat((head.unsqueeze_(1),etypes.unsqueeze_(1),tail.unsqueeze_(1)),1)
        return labels,triplets

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def forward(self, g, embed,graphs=None):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        labels,triplets=self.generate_samples(g)
        if labels is None:
            res=torch.tensor([0.0], requires_grad=True)
            if self.use_cuda:
                res=res.cuda()
            return res
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss
class LinkPredictorHomoLS(nn.Module):
    def __init__(self, h_dim, num_rels,use_cuda, reg_param=0):
        super(LinkPredictorHomoLS, self).__init__()
        self.reg_param = reg_param
        self.use_cuda=use_cuda
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score
    def generate_samples(self,pos_graph,neg_graph):
        # TODO move to dataloader so that it is faster

        etypes = pos_graph.edata[dgl.ETYPE]
        u, v = pos_graph.all_edges()

        nu, nv = neg_graph.all_edges()
        rate = int(len(nu) / len(u))
        netypes = etypes.repeat_interleave(rate)
        head=torch.cat((u,nu))
        tail=torch.cat((v,nv))
        etypes=torch.cat((etypes,netypes))
        labels=torch.zeros(len(head)).float()
        if self.use_cuda:
            labels=labels.cuda()
        labels[:len(u)]=1

        triplets=torch.cat((head.unsqueeze_(1),etypes.unsqueeze_(1),tail.unsqueeze_(1)),1)
        return labels,triplets

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def forward(self, g, embed,graphs):
        pos_graph,neg_graph=graphs
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        labels,triplets=self.generate_samples(pos_graph,neg_graph)
        if labels is None:
            res=torch.tensor([0.0], requires_grad=True)
            if self.use_cuda:
                res=res.cuda()
            return res
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

class LinkPredictor(nn.Module):
    def __init__(self, out_dim, etypes, ntype2id,reg_param=0,use_cuda=False,edg_pct=0.8,ng_rate=5,filtered=False,shared_rel_emb=False):
        super(LinkPredictor, self).__init__()
        self.reg_param = reg_param
        self.etypes=etypes

        self.ng_rate=ng_rate
        self.filtered = filtered
        self.ntype2id=ntype2id
        self.edg_pct=edg_pct
        self.use_cuda=use_cuda
        self.out_dim=out_dim
        self.shared_rel_emb=shared_rel_emb
        #self.params = nn.ParameterDict({ ename: nn.Parameter(torch.Tensor(out_dim,1))  for ename in self.etypes})
        if self.shared_rel_emb:
            w_relation = nn.Parameter(torch.Tensor(out_dim, 1))
            nn.init.xavier_uniform_(w_relation,
                                   gain=nn.init.calculate_gain('relu'))
            self.w_relation=w_relation
        else:
            self.w_relation = nn.ParameterDict()
            w_relation={}

            for ename in self.etypes:
                w_relation[ename] = nn.Parameter(torch.Tensor(out_dim,1))
                nn.init.xavier_uniform_(w_relation[ename],
                                    gain=nn.init.calculate_gain('relu'))
            self.w_relation.update(w_relation)
    def calc_pos_score_with_rids(self, h_emb, t_emb, rids,etypes2ids,device=None):
        # DistMult
        if self.shared_rel_emb:
            r = self.w_relation.repeat(( 1,len(rids))).T
        else:
            w_relation_mat = torch.zeros((len(self.w_relation), self.out_dim)).to(device)

            for etype in etypes2ids.keys():
                w_relation_mat[etypes2ids[etype], :] = self.w_relation[etype].squeeze()
            r = w_relation_mat[rids]


        score = torch.sum(h_emb * r * t_emb, dim=-1)
        return score

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size,etypes2ids, device=None):
        hidden_dim = heads.shape[1]
        if self.shared_rel_emb:
            r = self.w_relation.repeat(( 1,len(rids))).T
        else:
            w_relation_mat = torch.zeros((len(self.w_relation), self.out_dim)).to(device)

            for etype in etypes2ids.keys():
                w_relation_mat[etypes2ids[etype], :] = self.w_relation[etype].squeeze()
            r = w_relation_mat[rids]

        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = torch.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size,etypes2ids, device=None):
        hidden_dim = tails.shape[1]
        if self.shared_rel_emb:
            r = self.w_relation.repeat(( 1,len(rids))).T
        else:
            w_relation_mat = torch.zeros((len(self.w_relation), self.out_dim)).to(device)

            for etype in etypes2ids.keys():
                w_relation_mat[etypes2ids[etype], :] = self.w_relation[etype].squeeze()
            r = w_relation_mat[rids]
        if device is not None:
            r = r.to(device)
            heads = heads.to(device)
            tails = tails.to(device)
        heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
        heads = torch.transpose(heads, 1, 2)
        tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, heads)

    def calc_score(self, g,embedding, dict_s_d):
        # DistMult
        score={}

        for etype in self.etypes:
            (stype,e,dtype)=g.to_canonical_etype(etype)
            s = embedding[self.ntype2id[stype]][dict_s_d[etype][:, 0]]
            if self.shared_rel_emb:
                r = self.w_relation.repeat((1,len(s))).T
            else:
                r = self.w_relation[etype].squeeze()

            o = embedding[self.ntype2id[dtype]][dict_s_d[etype][:, 1]]
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def calc_score_mb(self, g,embedding, dict_s_d):
        # DistMult
        score = {}
        # TODO maybe represent triplets as three arrays to make it faster.
        for etype in dict_s_d.keys():
            (stype, e, dtype)=g.to_canonical_etype(etype)
            s = embedding[stype][dict_s_d[etype][0]]
            if self.shared_rel_emb:
                r = self.w_relation.repeat((1,len(s))).T
            else:
                r = self.w_relation[etype].squeeze()
            # TODO consider other formulations metapath2vec
            o = embedding[dtype][dict_s_d[etype][1]]
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
            loss=0
            for e in embedding:
                loss+=torch.mean(e.pow(2))
            if self.shared_rel_emb:
                loss += torch.mean(self.w_relation.pow(2))
            else:
                for e in self.w_relation.keys():
                    loss+=torch.mean(self.w_relation[e].pow(2))
            return loss

    def forward(self, g,embed, edict_s_d, e_dict_labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(g,embed, edict_s_d)
        predict_loss=0
        for etype in self.etypes:
            predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss(embed)

        return predict_loss + self.reg_param * reg_loss

    def generate_samples(self,g):
        # TODO move to dataloader so that it is faster
        edg_pct=self.edg_pct
        ng_rate=self.ng_rate
        filtered=self.filtered
        e_dict_labels={}
        edict_s_d={}
        for etype in g.etypes:
            u, v = g.all_edges(etype=etype)
            (srctype, ety, desttype) = g.to_canonical_etype(etype)
            # keep only pairs of nodes between dstnodes
            src_len = g.dstnodes[srctype].data['_ID'].shape[0]
            dest_len = g.dstnodes[desttype].data['_ID'].shape[0]
            src_nodes_in_dest = u < src_len
            if src_len==0 or dest_len==0 or torch.sum(src_nodes_in_dest)==0:
                continue
            else:
                pos_head = u[src_nodes_in_dest]
                pos_tail = v[src_nodes_in_dest]
                # filter for edg_pct
                pos_head = pos_head[:int(edg_pct * pos_head.shape[0])]
                pos_tail = pos_tail[:int(edg_pct * pos_tail.shape[0])]
                size_of_batch = pos_head.shape[0]
                labels = torch.zeros(size_of_batch * (ng_rate + 1)).float()
                if self.use_cuda:
                    labels=labels.cuda()
                labels[: size_of_batch] = 1
                head = pos_head.repeat((ng_rate + 1))
                tail = pos_tail.repeat((ng_rate + 1))
                # TODO filter edges
                #  set_pos_edges ={(1,3),(1,2)}
                #  set_pos_edges = {(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2),(1, 3), (1, 2)}
                #  neg_edges=set_neg_edges.minus(set_pos_edges)
                head[size_of_batch:] = torch.randint(src_len, (size_of_batch * ng_rate,))
                if filtered:
                    pos_pair = set(tuple(map(tuple, torch.cat((pos_head.unsqueeze(0), pos_tail.unsqueeze(0)), dim=0).transpose(1,0).cpu().numpy())))
                    neg_pair = set(tuple(map(tuple,torch.cat((head[size_of_batch:].unsqueeze(0),
                                          tail[size_of_batch:].unsqueeze(0)), dim=0).transpose(1,0).cpu().numpy())))
                    filt_neg_pair=neg_pair.difference(pos_pair)
                e_dict_labels[etype]=labels
                edict_s_d[etype]=(head,tail)
        return e_dict_labels,edict_s_d
    def regularization_loss_mb(self, embedding):
            loss = 0
            for ntype in embedding.keys():
                loss += torch.mean(embedding[ntype].pow(2))

            if self.shared_rel_emb:
                loss += torch.mean(self.w_relation.pow(2))
            else:
                for e in self.w_relation.keys():
                    loss += torch.mean(self.w_relation[e].pow(2))
            return loss
    def forward_mb(self, g,embed):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)

        e_dict_labels,edict_s_d=self.generate_samples(g=g)
        score = self.calc_score_mb(g,embed, edict_s_d)
        predict_loss=0
        for etype in e_dict_labels.keys():
            if len(score[etype])>0:
                predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

        reg_loss = self.regularization_loss_mb(embed)
        if self.reg_param==0:
            return predict_loss
        return predict_loss + self.reg_param * reg_loss
class AttributeDecoder(nn.Module):
    def __init__(self, h_dim, reconstruct_dim=1, use_cuda=False):
        super(AttributeDecoder, self).__init__()
        self.h_dim=h_dim
        self.reconstruct_dim=reconstruct_dim
        self.reconstruction_layer = nn.Linear(self.h_dim, self.reconstruct_dim)
    def forward(self, h):
        h=self.reconstruction_layer(h)
        return h

class NodeMotifDecoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dict, distribution=False,activation=nn.ReLU(), single_layer=False,output=True):
        '''

        :param out_dim:
        :param in_dim:
        :param h_dim:
        :param activation:
        '''

        super(NodeMotifDecoder, self).__init__()
        self.activation=activation
        self.h_dim=h_dim
        self.weight=nn.ModuleDict()
        self.single_layer=single_layer
        self.output=output
        self.distribution=distribution
        layers=[]
        for name in out_dict.keys():
            layers=[]
            if self.single_layer:
                layers.append(nn.Linear( in_dim,  out_dict[name],bias=False))
            else:
                layers.append(nn.Linear( in_dim, self.h_dim))
                layers.append(activation)

                layers.append(nn.Linear( self.h_dim, out_dict[name],bias=False))
            #if not self.distribution:
            #    layers.append(nn.Sigmoid())
            #make sure is correct
            self.weight[name]=nn.Sequential(*layers)

    def forward(self, g, h):
        node_embed=h
        loss=0
        #g = g.local_var()
        train_acc=0
        train_acc_auc=0

        for name in g.dsttypes:
            if name in node_embed and node_embed[name].shape[0]>0:
                reconstructed=self.weight[name](node_embed[name])
                if self.distribution:
                    loss += F.mse_loss(reconstructed, g.dstnodes[name].data['motifs'])
                else:
                    lbl=g.dstnodes[name].data['motifs']
                    logits=reconstructed
                    loss += F.binary_cross_entropy_with_logits(logits, lbl)
                    if self.output:
                        train_acc += torch.sum(logits.argmax(dim=1) == lbl.argmax(dim=1)).item() / logits.shape[0]
                        pred = torch.sigmoid(logits).detach().cpu().numpy()
                        try:
                            train_acc_auc += roc_auc_score(lbl.cpu().numpy(), pred)
                        except ValueError:
                            pass
        if not self.distribution and self.output:
                print('Motif prediction accuracy: {:.4f} | AUC: {:.4f}'.
                  format(train_acc/len(g.dsttypes),train_acc_auc/len(g.dsttypes)))
            #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]

        return loss
class NodeClassifierRGCN(nn.Module):
    def __init__(self,  in_dim, out_dim, rel_names,num_bases,
                  activation=partial(F.softmax, dim=1), use_self_loop=False):

        super(NodeClassifierRGCN, self).__init__()

        self.layer=RelGraphConvHetero(in_dim, out_dim, rel_names, "basis", num_bases,activation=activation,
                 self_loop=use_self_loop)

    def forward(self,g,h_d):
        h = self.layer(g, h_d)
        return h
    def forward_mb(self,g, h_d):
        h=self.layer.forward_mb(g, h_d)
        return h
class MultipleAttributeDecoder(nn.Module):
    def __init__(self, out_size_dict, in_size, h_dim, masked_node_types,
                 loss_over_all_nodes,activation=nn.ReLU(),single_layer=False,use_cluster=False,output=True):
        '''

        :param out_size_dict:
        :param in_size:
        :param h_dim:
        :param masked_node_types: Node types to not penalize for reconstruction
        :param activation:
        '''

        super(MultipleAttributeDecoder, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim

        self.masked_node_types=masked_node_types
        self.loss_over_all_nodes=loss_over_all_nodes
        self.single_layer=single_layer
        self.use_cluster=use_cluster
        self.output=output
        self.weight = nn.ModuleDict()
        for name in out_size_dict.keys():
            layers=[]
            if self.single_layer:
                layers.append(nn.Linear(in_size, out_size_dict[name], bias=False))
            else:
                layers.append(nn.Linear( in_size, self.h_dim))
                layers.append(activation)
                layers.append(nn.Linear(self.h_dim, out_size_dict[name], bias=False))
            if use_cluster:
                layers.append(nn.Sigmoid())
            self.weight[name]=nn.Sequential(*layers)

    def loss_function(self,pred,act):
        if self.use_cluster:
            loss=F.cross_entropy(pred,torch.argmax(act,dim=1))
        else:
            loss=F.mse_loss(pred,act)
        return loss

    def forward(self,G,h,masked_nodes):
        g = G#.local_var()
        loss=0
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = h[i]
        for name in self.weight:
            if name not in self.masked_node_types:
                g.nodes[name].data['h']=self.weight[name](g.nodes[name].data['x'])
                if bool(masked_nodes):
                    if not self.loss_over_all_nodes:
                        loss+=self.loss_function(g.nodes[name].data['h'][masked_nodes[name]],g.nodes[name].data['masked_values'][masked_nodes[name]])
                    else:
                        loss += self.loss_function(g.nodes[name].data['h'],
                                           g.nodes[name].data['masked_values'])

                else:
                    loss += self.loss_function(g.nodes[name].data['h'],
                                       g.nodes[name].data['h_f'])
        #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]

        return loss


    def forward_mb(self, g, h, masked_nodes):
        node_embed=h
        loss=0
        #g = g.local_var()
        if self.use_cluster:
            data_param='h_clusters'
        else:
            data_param="h_f"
        total=0
        correct=0
        for name in self.weight:
            if g.dstnodes[name].data.get(data_param, None) is not None and node_embed[name].shape[0]>0:
                reconstructed=self.weight[name](node_embed[name])
                if bool(masked_nodes):
                    if not self.loss_over_all_nodes:
                        loss+=self.loss_function(reconstructed[masked_nodes[name]],g.dstnodes[name].data[data_param])
                    else:
                        loss += self.loss_function(reconstructed, g.dstnodes[name].data[data_param])

                else:
                    loss += self.loss_function(reconstructed, g.dstnodes[name].data[data_param])
                if self.use_cluster and self.output:
                    _, predicted = torch.max(reconstructed.data, 1)
                    labels = torch.argmax(g.dstnodes[name].data[data_param], dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
        if self.use_cluster and self.output:
            print('Cluster accuracy: %d %%' % (
                    100 * correct / total))
        #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]


        return loss
class ClusterRecoverDecoderHomo(nn.Module):
    def __init__(self, n_cluster, in_size, h_dim,
 activation=nn.ReLU(), single_layer=True, output_cluster_accuracy=True):
        '''

        :param out_size_dict:
        :param in_size:
        :param h_dim:
        :param masked_node_types: Node types to not penalize for reconstruction
        :param activation:
        '''

        super(ClusterRecoverDecoderHomo, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim
        self.n_cluster=n_cluster

        self.single_layer=single_layer
        self.output_cluster_accuracy=output_cluster_accuracy
        layers=[]
        if self.single_layer:
                layers.append(nn.Linear(in_size, n_cluster, bias=False))
        else:
                layers.append(nn.Linear( in_size, self.h_dim))
                layers.append(activation)
                layers.append(nn.Linear(self.h_dim, n_cluster, bias=False))
        #if use_cluster:
        #    layers.append(nn.Sigmoid())
        self.weight=nn.Sequential(*layers)

    def loss_function(self,pred,act):
        loss=F.cross_entropy(pred,torch.argmax(act,dim=1))
        return loss


    def forward(self, h,cluster_assignments,node_ids, node_tids, type_ids):
        """Forward computation
        Parameters
        ----------
        node_ids : tensor
            node ids to generate embedding for.
        node_ids : tensor
            node type ids
        Returns
        -------
        tensor
            embeddings as the input of the next layer
        """
        #tsd_ids = node_ids.to(self.weight.device)
        idx = torch.empty(node_ids.shape[0], dtype=torch.int64, device=h.device)
        clusters_assignments_mapped = []
        num_nodes = 0
        for ntype in range(len(cluster_assignments)):
            if cluster_assignments[ntype] is not None:
                loc = node_tids == ntype
                clusters_assignments_mapped.append(cluster_assignments[ntype][type_ids[loc]].to(h.device) )
            else:
                loc = node_tids == ntype
                clusters_assignments_mapped.append(torch.zeros((sum(loc),self.n_cluster)).to(h.device))
            idx[loc] = torch.arange(len(clusters_assignments_mapped[-1]), device=h.device) + num_nodes
            num_nodes += len(clusters_assignments_mapped[-1])
        clusters_assignments_mapped = torch.cat(clusters_assignments_mapped)

        cluster_assignments=clusters_assignments_mapped[idx]
        ##
        node_embed=h
        loss=0
        #g = g.local_var()

        total=0
        correct=0
        reconstructed=self.weight(node_embed)

        non_empty_mask = cluster_assignments.abs().sum(dim=1).bool()
        cluster_assignments=cluster_assignments[non_empty_mask,:]
        reconstructed=reconstructed[non_empty_mask,:]
        loss += self.loss_function(reconstructed, cluster_assignments)

        if self.output_cluster_accuracy:
                    _, predicted = torch.max(reconstructed.data, 1)
                    labels = torch.argmax(cluster_assignments, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    print('Cluster accuracy: %d %%' % (
                    100 * correct / total))
        #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]


        return loss

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features
class MutualInformationDiscriminatorHomo(nn.Module):
    # returns the MI loss function follows the dgl implementation
    def __init__(self, n_hidden,average_across_node_types=True,convex_combination_weight=0.9):
        super(MutualInformationDiscriminatorHomo, self).__init__()
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        self.average_across_node_types=average_across_node_types
        self.convex_combination_weight=convex_combination_weight
        self.global_summary=None
        # keep a global summary
        #self.positives


    def forward(self, positives, negatives):
        l1=0
        l2=0
        if self.average_across_node_types:
            summary = torch.sigmoid(positives.mean(dim=0))
            if self.convex_combination_weight is not None:
                if self.global_summary is not None :
                    self.global_summary= \
                        self.convex_combination_weight*self.global_summary.detach()+(1-self.convex_combination_weight)*summary
                    summary=self.global_summary
                else:
                    self.global_summary=summary

            positive = self.discriminator(positives.mean(dim=0), summary)
            negative = self.discriminator(negatives.mean(dim=0), summary)
            l1 += self.loss(positive, torch.ones_like(positive))
            l2 += self.loss(negative, torch.zeros_like(negative))
            return l1 + l2
        else:
            raise NotImplementedError

class MutualInformationDiscriminator(nn.Module):
    # returns the MI loss function follows the dgl implementation
    def __init__(self, n_hidden,average_across_node_types=False,focus_category=None):
        super(MutualInformationDiscriminator, self).__init__()
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        self.average_across_node_types=average_across_node_types
        self.focus_category=focus_category

        # keep a global summary
        #self.positives


    def forward(self, positives, negatives):
        l1=0
        l2=0
        if self.average_across_node_types:
            positives=torch.cat(positives,dim=0)
            negatives=torch.cat(negatives,dim=0)
            summary = torch.sigmoid(positives.mean(dim=0))
            positive = self.discriminator(positives.mean(dim=0), summary)
            negative = self.discriminator(negatives.mean(dim=0), summary)
            l1 += self.loss(positive, torch.ones_like(positive))
            l2 += self.loss(negative, torch.zeros_like(negative))

            return l1 + l2
        else:
            for positive,negative in zip(positives,negatives):
                summary = torch.sigmoid(positive.mean(dim=0))

                positive = self.discriminator(positive, summary)
                negative = self.discriminator(negative, summary)

                l1 += self.loss(positive, torch.ones_like(positive))
                l2 += self.loss(negative, torch.zeros_like(negative))
            return l1+l2

    def forward_mb(self, positives, negatives):
        l1=0
        l2=0
        # TODO summary per node type or across all node types? for infomax
        if self.focus_category is not None:
            if self.focus_category in positives.keys():
                positive = positives[self.focus_category ]
                negative = negatives[self.focus_category ]
                summary = torch.sigmoid(positive.mean(dim=0))

                positive = self.discriminator(positive, summary)
                negative = self.discriminator(negative, summary)

                l1 += self.loss(positive, torch.ones_like(positive))
                l2 += self.loss(negative, torch.zeros_like(negative))
                return l1 + l2

        for ntype in positives.keys():
            if positives[ntype].shape[0]>0:
                positive=positives[ntype]
                negative=negatives[ntype]
                summary = torch.sigmoid(positive.mean(dim=0))

                positive = self.discriminator(positive, summary)
                negative = self.discriminator(negative, summary)

                l1 += self.loss(positive, torch.ones_like(positive))
                l2 += self.loss(negative, torch.zeros_like(negative))
        return l1+l2