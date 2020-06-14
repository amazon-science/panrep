"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from node_sampling_masking import  HeteroNeighborSampler,InfomaxNodeRecNeighborSampler,LinkPredictorEvalSampler
import os
from utils import calculate_entropy
from datetime import datetime
import pickle
import argparse
import numpy as np
import copy
from utils import calc_mrr_pr
import time
import torch
import torch.nn.functional as F
import dgl
import torch as th
from classifiers import ClassifierMLP,DLinkPredictorOnlyRel
from load_data import load_univ_hetero_data,generate_rwalks
from model import PanRepHetero
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from node_supervision_tasks import MetapathRWalkerSupervision, LinkPredictor,LinkPredictorLearnableEmbed
from encoders import EncoderRelGraphConvHetero,EncoderRelGraphAttentionHetero
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
def evaluation_link_prediction_wembeds(test_g,model, embeddings,train_edges,valid_edges,test_edges,dim_size,eval_neg_cnt,n_layers,device):
    def transform_triplets(train_edges,etype2id,ntype2id):
        train_src = None
        # TODO have to map the etype and ntype to their integer ids.
        for key in train_edges.keys():
            if train_src is None:
                train_src = train_edges[key][0]
                train_dst = train_edges[key][1]
                train_rel = th.tensor(etype2id[key[1]]).repeat((train_src.shape[0]))
                train_src_type = th.tensor(ntype2id[key[0]]).repeat((train_src.shape[0]))
                train_dst_type = th.tensor(ntype2id[key[2]]).repeat((train_src.shape[0]))
            else:
                train_src = torch.cat((train_src, train_edges[key][0]))
                train_dst = torch.cat((train_dst, train_edges[key][1]))
                train_rel = torch.cat((train_rel, th.tensor(etype2id[key[1]]).repeat((train_edges[key][0].shape[0]))))
                train_src_type = torch.cat(
                    (train_src_type, th.tensor(ntype2id[key[0]]).repeat((train_edges[key][0].shape[0]))))
                train_dst_type = torch.cat(
                    (train_dst_type, th.tensor(ntype2id[key[2]]).repeat((train_edges[key][0].shape[0]))))
        perm=torch.randperm(train_src.shape[0])
        train_src=train_src[perm]
        train_dst = train_dst[perm]
        train_src_type = train_src_type[perm]
        train_rel = train_rel[perm]
        train_dst_type = train_dst_type[perm]

        return (train_src,train_dst,train_src_type,train_rel,train_dst_type)
    def prepare_triplets(train_data, valid_data, test_data):
        if len(train_data) == 3:
            train_src, train_rel, train_dst = train_data
            train_htypes = None
            train_ttypes = None
        else:
            assert len(train_data) == 5
            train_src, train_dst, train_src_type, train_rel, train_dst_type = train_data
            train_htypes = (train_src_type)
            train_ttypes = (train_dst_type)
        head_ids = (train_src)
        tail_ids = (train_dst)
        etypes = (train_rel)
        num_train_edges = etypes.shape[0]
        # pos_seed = th.arange(batch_size * 5000) #num_train_edges//batch_size) * batch_size)
        if len(valid_data) == 3:
            valid_src, valid_rel, valid_dst = valid_data
            valid_htypes = None
            valid_ttypes = None
            valid_neg_htypes = None
            valid_neg_ttypes = None
        else:
            assert len(valid_data) == 5
            valid_src, valid_dst, valid_src_trype, valid_rel, valid_dst_type = valid_data
            valid_htypes = (valid_src_trype)
            valid_ttypes = (valid_dst_type)
            valid_neg_htypes = th.cat([train_htypes, valid_htypes])
            valid_neg_ttypes = th.cat([train_ttypes, valid_ttypes])
        valid_head_ids = (valid_src)
        valid_tail_ids = (valid_dst)
        valid_etypes = (valid_rel)
        valid_neg_head_ids = th.cat([head_ids, valid_head_ids])
        valid_neg_tail_ids = th.cat([tail_ids, valid_tail_ids])
        valid_neg_etypes = th.cat([etypes, valid_etypes])
        num_valid_edges = valid_etypes.shape[0] + num_train_edges
        valid_seed = th.arange(valid_etypes.shape[0])
        if len(test_data) == 3:
            test_src, test_rel, test_dst = test_data
            test_htypes = None
            test_ttypes = None
            test_neg_htypes = None
            test_neg_ttypes = None
        else:
            assert len(test_data) == 5
            test_src, test_dst, test_src_type, test_rel, test_dst_type = test_data
            test_htypes = (test_src_type)
            test_ttypes = (test_dst_type)
            test_neg_htypes = th.cat([valid_neg_htypes, test_htypes])
            test_neg_ttypes = th.cat([valid_neg_ttypes, test_ttypes])
        test_head_ids = (test_src)
        test_tail_ids = (test_dst)
        test_etypes = (test_rel)
        test_neg_head_ids = th.cat([valid_neg_head_ids, test_head_ids])
        test_neg_tail_ids = th.cat([valid_neg_tail_ids, test_tail_ids])
        test_neg_etypes = th.cat([valid_neg_etypes, test_etypes])
        pos_pairs = (test_head_ids, test_etypes, test_tail_ids, test_htypes, test_ttypes)
        neg_pairs = (test_neg_head_ids, test_neg_etypes, test_neg_tail_ids, test_neg_htypes, test_neg_ttypes)
        return pos_pairs, neg_pairs

    def creat_eval_minibatch(test_g, n_layers):
        eval_minibatch_blocks = []
        eval_minibatch_info = []
        for ntype in test_g.ntypes:
            n_nodes = test_g.number_of_nodes(ntype)
            eval_minibatch = 512
            for i in range(int((n_nodes + eval_minibatch - 1) // eval_minibatch)):
                cur = {}
                valid_blocks = []
                cur[ntype] = th.arange(i * eval_minibatch,
                                       (i + 1) * eval_minibatch \
                                           if (i + 1) * eval_minibatch < n_nodes \
                                           else n_nodes)
                # record the seed
                eval_minibatch_info.append((ntype, cur[ntype]))
                for _ in range(n_layers):
                    #print(cur)
                    frontier = dgl.in_subgraph(test_g, cur)

                    block = dgl.to_block(frontier, cur)

                    cur = {}
                    for s_ntype in block.srctypes:
                        cur[s_ntype] = block.srcnodes[s_ntype].data[dgl.NID]
                    block=block.to(device)
                    valid_blocks.insert(0, block)

                eval_minibatch_blocks.append(valid_blocks)
        for i in range(len(eval_minibatch_blocks)):
                for ntype in eval_minibatch_blocks[i][0].ntypes:
                    if eval_minibatch_blocks[i][0].number_of_src_nodes(ntype)>0:
                        if test_g.nodes[ntype].data.get("h_f", None) is not None:
                            eval_minibatch_blocks[i][0].srcnodes[ntype].data['h_f'] = test_g.nodes[ntype].data['h_f'][
                                eval_minibatch_blocks[i][0].srcnodes[ntype].data['_ID']].to(device)
        return eval_minibatch_info, eval_minibatch_blocks

    def fullgraph_eval(eval_g, model,embeddings, device, dim_size,
                       pos_pairs, neg_pairs, eval_neg_cnt,ntype2id,etype2id):
        model.eval()
        t0 = time.time()

        p_h = embeddings
        with th.no_grad():

            test_head_ids, test_etypes, test_tail_ids, test_htypes, test_ttypes = pos_pairs
            test_neg_head_ids, _, test_neg_tail_ids, test_neg_htypes, test_neg_ttypes = neg_pairs

            mrr = 0
            mr = 0
            hit1 = 0
            hit3 = 0
            hit10 = 0
            pos_batch_size = 1000
            pos_cnt = test_head_ids.shape[0]
            total_cnt = 0

            # unique test head and tail nodes
            if test_htypes is None:
                unique_neg_head_ids = th.unique(test_neg_head_ids)
                unique_neg_tail_ids = th.unique(test_neg_tail_ids)
                unique_neg_htypes = None
                unique_neg_ttypes = None
            else:
                unique_neg_head_ids = []
                unique_neg_tail_ids = []
                unique_neg_htypes = []
                unique_neg_ttypes = []
                for nt in eval_g.ntypes:
                    cols = (test_neg_htypes == ntype2id[nt])
                    unique_ids = th.unique(test_neg_head_ids[cols])
                    unique_neg_head_ids.append(unique_ids)
                    unique_neg_htypes.append(th.full((unique_ids.shape[0],),  ntype2id[nt]))

                    cols = (test_neg_ttypes ==  ntype2id[nt])
                    unique_ids = th.unique(test_neg_tail_ids[cols])
                    unique_neg_tail_ids.append(unique_ids)
                    unique_neg_ttypes.append(th.full((unique_ids.shape[0],),  ntype2id[nt]))
                unique_neg_head_ids = th.cat(unique_neg_head_ids)
                unique_neg_tail_ids = th.cat(unique_neg_tail_ids)
                unique_neg_htypes = th.cat(unique_neg_htypes)
                unique_neg_ttypes = th.cat(unique_neg_ttypes)

            if eval_neg_cnt > 0:
                total_neg_head_seed = th.randint(unique_neg_head_ids.shape[0],
                                                 (eval_neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
                total_neg_tail_seed = th.randint(unique_neg_tail_ids.shape[0],
                                                 (eval_neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
            for p_i in range(int((pos_cnt + pos_batch_size - 1) // pos_batch_size)):
                print("Eval {}-{}".format(p_i * pos_batch_size,
                                          (p_i + 1) * pos_batch_size \
                                              if (p_i + 1) * pos_batch_size < pos_cnt \
                                              else pos_cnt))
                sub_test_head_ids = test_head_ids[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                sub_test_etypes = test_etypes[p_i * pos_batch_size: \
                                              (p_i + 1) * pos_batch_size \
                                                  if (p_i + 1) * pos_batch_size < pos_cnt \
                                                  else pos_cnt]
                sub_test_tail_ids = test_tail_ids[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]

                if test_htypes is None:
                    phead_emb = p_h['node'][sub_test_head_ids]
                    ptail_emb = p_h['node'][sub_test_tail_ids]
                else:
                    sub_test_htypes = test_htypes[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                    sub_test_ttypes = test_ttypes[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                    phead_emb = th.empty((sub_test_head_ids.shape[0], dim_size), device=device)
                    ptail_emb = th.empty((sub_test_tail_ids.shape[0], dim_size), device=device)
                    for nt in eval_g.ntypes:
                        if nt in p_h:
                            loc = (sub_test_htypes ==  ntype2id[nt])
                            phead_emb[loc] = p_h[nt][sub_test_head_ids[loc]]
                            loc = (sub_test_ttypes ==  ntype2id[nt])
                            ptail_emb[loc] = p_h[nt][sub_test_tail_ids[loc]]

                pos_scores = model.calc_pos_score_with_rids(phead_emb, ptail_emb, sub_test_etypes,etype2id,device)
                pos_scores = F.logsigmoid(pos_scores).reshape(phead_emb.shape[0], -1).detach().cpu()

                if eval_neg_cnt > 0:
                    neg_head_seed = total_neg_head_seed[p_i * eval_neg_cnt:(p_i + 1) * eval_neg_cnt]
                    neg_tail_seed = total_neg_tail_seed[p_i * eval_neg_cnt:(p_i + 1) * eval_neg_cnt]
                    seed_test_neg_head_ids = unique_neg_head_ids[neg_head_seed]
                    seed_test_neg_tail_ids = unique_neg_tail_ids[neg_tail_seed]
                    if test_neg_htypes is not None:
                        seed_test_neg_htypes = unique_neg_htypes[neg_head_seed]
                        seed_test_neg_ttypes = unique_neg_ttypes[neg_tail_seed]
                else:
                        seed_test_neg_head_ids = unique_neg_head_ids
                        seed_test_neg_tail_ids = unique_neg_tail_ids
                        seed_test_neg_htypes = unique_neg_htypes
                        seed_test_neg_ttypes = unique_neg_ttypes

                neg_batch_size = 10000
                head_neg_cnt = seed_test_neg_head_ids.shape[0]
                tail_neg_cnt = seed_test_neg_tail_ids.shape[0]
                t_neg_score = []
                h_neg_score = []
                for n_i in range(int((head_neg_cnt + neg_batch_size - 1) // neg_batch_size)):
                    sub_test_neg_head_ids = seed_test_neg_head_ids[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < head_neg_cnt
                                                                   else head_neg_cnt]
                    if test_htypes is None:
                        nhead_emb = p_h['node'][sub_test_neg_head_ids]
                    else:
                        sub_test_neg_htypes = seed_test_neg_htypes[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < head_neg_cnt
                                                                   else head_neg_cnt]

                        nhead_emb = th.empty((sub_test_neg_head_ids.shape[0], dim_size), device=device)
                        for nt in eval_g.ntypes:
                            if nt in p_h:
                                loc = (sub_test_neg_htypes ==  ntype2id[nt])
                                nhead_emb[loc] = p_h[nt][sub_test_neg_head_ids[loc]]

                    h_neg_score.append(
                        model.calc_neg_head_score(nhead_emb,
                                                                 ptail_emb,
                                                                 sub_test_etypes,
                                                                 1,
                                                                 ptail_emb.shape[0],
                                                                 nhead_emb.shape[0],etype2id,device).reshape(-1, nhead_emb.shape[
                        0]).detach().cpu())

                for n_i in range(int((tail_neg_cnt + neg_batch_size - 1) // neg_batch_size)):
                    sub_test_neg_tail_ids = seed_test_neg_tail_ids[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < tail_neg_cnt
                                                                   else tail_neg_cnt]

                    if test_htypes is None:
                        ntail_emb = p_h['node'][sub_test_neg_tail_ids]
                    else:
                        sub_test_neg_ttypes = seed_test_neg_ttypes[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < tail_neg_cnt
                                                                   else tail_neg_cnt]
                        ntail_emb = th.empty((sub_test_neg_tail_ids.shape[0], dim_size), device=device)
                        for nt in eval_g.ntypes:
                            if nt in p_h:
                                loc = (sub_test_neg_ttypes ==  ntype2id[nt])
                                ntail_emb[loc] = p_h[nt][sub_test_neg_tail_ids[loc]]

                    t_neg_score.append(model.calc_neg_tail_score(phead_emb,
                                                                 ntail_emb,
                                                                 sub_test_etypes,
                                                                 1,
                                                                 phead_emb.shape[0],
                                                                 ntail_emb.shape[0],etype2id,device).reshape(-1, ntail_emb.shape[
                        0]).detach().cpu())
                t_neg_score = th.cat(t_neg_score, dim=1)
                h_neg_score = th.cat(h_neg_score, dim=1)
                t_neg_score = F.logsigmoid(t_neg_score)
                h_neg_score = F.logsigmoid(h_neg_score)

                canonical_etypes = eval_g.canonical_etypes
                for idx in range(phead_emb.shape[0]):
                    if test_htypes is None:
                        tail_pos = eval_g.has_edges_between(
                            th.full((seed_test_neg_tail_ids.shape[0],), sub_test_head_ids[idx]).long(),
                            seed_test_neg_tail_ids,
                            etype=test_g.etypes[(sub_test_etypes[idx].numpy().item())])
                        head_pos = eval_g.has_edges_between(seed_test_neg_head_ids,
                                                            th.full((seed_test_neg_head_ids.shape[0],),
                                                                    sub_test_tail_ids[idx]).long(),
                                                            etype=test_g.etypes[(sub_test_etypes[idx].numpy().item())])
                        loc = tail_pos == 1
                        t_neg_score[idx][loc] += pos_scores[idx]
                        loc = head_pos == 1
                        h_neg_score[idx][loc] += pos_scores[idx]

                    else:
                        head_type = test_g.ntypes[(sub_test_htypes[idx].numpy())]
                        tail_type = test_g.ntypes[(sub_test_ttypes[idx].numpy())]

                        for t in eval_g.ntypes:
                            if (head_type, test_g.etypes[(sub_test_etypes[idx].numpy().item())], t) in canonical_etypes:
                                loc = (seed_test_neg_ttypes ==  ntype2id[t])
                                t_neg_tail_ids = seed_test_neg_tail_ids[loc]

                                # there is some neg tail in this type
                                if t_neg_tail_ids.shape[0] > 0:
                                    tail_pos = eval_g.has_edges_between(
                                        th.full((t_neg_tail_ids.shape[0],), sub_test_head_ids[idx]).long(),
                                        t_neg_tail_ids,
                                        etype=(head_type,
                                               test_g.etypes[(sub_test_etypes[idx].numpy().item())],
                                               t))
                                    t_neg_score[idx][loc][tail_pos == 1] += pos_scores[idx]
                            if (t, test_g.etypes[(sub_test_etypes[idx].numpy().item())], tail_type) in canonical_etypes:
                                loc = (seed_test_neg_htypes == ntype2id[t])
                                t_neg_head_ids = seed_test_neg_head_ids[loc]

                                # there is some neg head in this type
                                if t_neg_head_ids.shape[0] > 0:
                                    head_pos = eval_g.has_edges_between(t_neg_head_ids,
                                                                        th.full((t_neg_head_ids.shape[0],),
                                                                                sub_test_tail_ids[idx]).long(),
                                                                        etype=(t,
                                                                               test_g.etypes[(sub_test_etypes[idx].numpy().item())]
                                                                               ,
                                                                               tail_type))
                                    h_neg_score[idx][loc][head_pos == 1] += pos_scores[idx]
                neg_score = th.cat([h_neg_score, t_neg_score], dim=1)

                rankings = th.sum(neg_score >= pos_scores, dim=1) + 1
                rankings = rankings.cpu().detach().numpy()
                for ranking in rankings:
                    mrr += 1.0 / ranking
                    mr += float(ranking)
                    hit1 += 1.0 if ranking <= 1 else 0.0
                    hit3 += 1.0 if ranking <= 3 else 0.0
                    hit10 += 1.0 if ranking <= 10 else 0.0
                    total_cnt += 1
        res="MRR {}\nMR {}\nHITS@1 {}\nHITS@3 {}\nHITS@10 {}".format(mrr / total_cnt,
                                                                       mr / total_cnt,
                                                                       hit1 / total_cnt,
                                                                       hit3 / total_cnt,
                                                                       hit10 / total_cnt)
        print("MRR {}\nMR {}\nHITS@1 {}\nHITS@3 {}\nHITS@10 {}".format(mrr / total_cnt,
                                                                       mr / total_cnt,
                                                                       hit1 / total_cnt,
                                                                       hit3 / total_cnt,
                                                                       hit10 / total_cnt))
        t1 = time.time()
        print("Full eval {} exmpales takes {} seconds".format(pos_scores.shape[0], t1 - t0))
        return res

    ntype2id = {}
    for i, ntype in enumerate(test_g.ntypes):
            ntype2id[ntype] = i
    etype2id = {}
    for i, etype in enumerate(test_g.etypes):
            etype2id[etype] = i
    train_data=transform_triplets(train_edges, etype2id, ntype2id)
    valid_data = transform_triplets(valid_edges, etype2id, ntype2id)
    test_data = transform_triplets(test_edges, etype2id, ntype2id)
    pos_pairs, neg_pairs=prepare_triplets(train_data, valid_data, test_data)
    #minibatch_info, minibatch_blocks=creat_eval_minibatch(test_g, n_layers)
    res=fullgraph_eval(test_g, model,embeddings, device, dim_size,
                   pos_pairs, neg_pairs, eval_neg_cnt,ntype2id,etype2id)
    return res

def evaluation_link_prediction(test_g,model,train_edges,valid_edges,test_edges,dim_size,eval_neg_cnt,n_layers,device):
    def transform_triplets(train_edges,etype2id,ntype2id):
        train_src = None
        # TODO have to map the etype and ntype to their integer ids.
        for key in train_edges.keys():
            if train_src is None:
                train_src = train_edges[key][0]
                train_dst = train_edges[key][1]
                train_rel = th.tensor(etype2id[key[1]]).repeat((train_src.shape[0]))
                train_src_type = th.tensor(ntype2id[key[0]]).repeat((train_src.shape[0]))
                train_dst_type = th.tensor(ntype2id[key[2]]).repeat((train_src.shape[0]))
            else:
                train_src = torch.cat((train_src, train_edges[key][0]))
                train_dst = torch.cat((train_dst, train_edges[key][1]))
                train_rel = torch.cat((train_rel, th.tensor(etype2id[key[1]]).repeat((train_edges[key][0].shape[0]))))
                train_src_type = torch.cat(
                    (train_src_type, th.tensor(ntype2id[key[0]]).repeat((train_edges[key][0].shape[0]))))
                train_dst_type = torch.cat(
                    (train_dst_type, th.tensor(ntype2id[key[2]]).repeat((train_edges[key][0].shape[0]))))
        perm=torch.randperm(train_src.shape[0])
        train_src=train_src[perm]
        train_dst = train_dst[perm]
        train_src_type = train_src_type[perm]
        train_rel = train_rel[perm]
        train_dst_type = train_dst_type[perm]

        return (train_src,train_dst,train_src_type,train_rel,train_dst_type)
    def prepare_triplets(train_data, valid_data, test_data):
        if len(train_data) == 3:
            train_src, train_rel, train_dst = train_data
            train_htypes = None
            train_ttypes = None
        else:
            assert len(train_data) == 5
            train_src, train_dst, train_src_type, train_rel, train_dst_type = train_data
            train_htypes = (train_src_type)
            train_ttypes = (train_dst_type)
        head_ids = (train_src)
        tail_ids = (train_dst)
        etypes = (train_rel)
        num_train_edges = etypes.shape[0]
        # pos_seed = th.arange(batch_size * 5000) #num_train_edges//batch_size) * batch_size)
        if len(valid_data) == 3:
            valid_src, valid_rel, valid_dst = valid_data
            valid_htypes = None
            valid_ttypes = None
            valid_neg_htypes = None
            valid_neg_ttypes = None
        else:
            assert len(valid_data) == 5
            valid_src, valid_dst, valid_src_trype, valid_rel, valid_dst_type = valid_data
            valid_htypes = (valid_src_trype)
            valid_ttypes = (valid_dst_type)
            valid_neg_htypes = th.cat([train_htypes, valid_htypes])
            valid_neg_ttypes = th.cat([train_ttypes, valid_ttypes])
        valid_head_ids = (valid_src)
        valid_tail_ids = (valid_dst)
        valid_etypes = (valid_rel)
        valid_neg_head_ids = th.cat([head_ids, valid_head_ids])
        valid_neg_tail_ids = th.cat([tail_ids, valid_tail_ids])
        valid_neg_etypes = th.cat([etypes, valid_etypes])
        num_valid_edges = valid_etypes.shape[0] + num_train_edges
        valid_seed = th.arange(valid_etypes.shape[0])
        if len(test_data) == 3:
            test_src, test_rel, test_dst = test_data
            test_htypes = None
            test_ttypes = None
            test_neg_htypes = None
            test_neg_ttypes = None
        else:
            assert len(test_data) == 5
            test_src, test_dst, test_src_type, test_rel, test_dst_type = test_data
            test_htypes = (test_src_type)
            test_ttypes = (test_dst_type)
            test_neg_htypes = th.cat([valid_neg_htypes, test_htypes])
            test_neg_ttypes = th.cat([valid_neg_ttypes, test_ttypes])
        test_head_ids = (test_src)
        test_tail_ids = (test_dst)
        test_etypes = (test_rel)
        test_neg_head_ids = th.cat([valid_neg_head_ids, test_head_ids])
        test_neg_tail_ids = th.cat([valid_neg_tail_ids, test_tail_ids])
        test_neg_etypes = th.cat([valid_neg_etypes, test_etypes])
        pos_pairs = (test_head_ids, test_etypes, test_tail_ids, test_htypes, test_ttypes)
        neg_pairs = (test_neg_head_ids, test_neg_etypes, test_neg_tail_ids, test_neg_htypes, test_neg_ttypes)
        return pos_pairs, neg_pairs

    def creat_eval_minibatch(test_g, n_layers):
        eval_minibatch_blocks = []
        eval_minibatch_info = []
        for ntype in test_g.ntypes:
            n_nodes = test_g.number_of_nodes(ntype)
            eval_minibatch = 512
            for i in range(int((n_nodes + eval_minibatch - 1) // eval_minibatch)):
                cur = {}
                valid_blocks = []
                cur[ntype] = th.arange(i * eval_minibatch,
                                       (i + 1) * eval_minibatch \
                                           if (i + 1) * eval_minibatch < n_nodes \
                                           else n_nodes)
                # record the seed
                eval_minibatch_info.append((ntype, cur[ntype]))
                for _ in range(n_layers):
                    #print(cur)
                    frontier = dgl.in_subgraph(test_g, cur)

                    block = dgl.to_block(frontier, cur)

                    cur = {}
                    for s_ntype in block.srctypes:
                        cur[s_ntype] = block.srcnodes[s_ntype].data[dgl.NID]
                    block=block.to(device)
                    valid_blocks.insert(0, block)

                eval_minibatch_blocks.append(valid_blocks)
        for i in range(len(eval_minibatch_blocks)):
                for ntype in eval_minibatch_blocks[i][0].ntypes:
                    if eval_minibatch_blocks[i][0].number_of_src_nodes(ntype)>0:
                        if test_g.nodes[ntype].data.get("h_f", None) is not None:
                            eval_minibatch_blocks[i][0].srcnodes[ntype].data['h_f'] = test_g.nodes[ntype].data['h_f'][
                                eval_minibatch_blocks[i][0].srcnodes[ntype].data['_ID']].to(device)
        return eval_minibatch_info, eval_minibatch_blocks

    def fullgraph_eval(eval_g, model, device, dim_size, minibatch_blocks, minibatch_info,
                       pos_pairs, neg_pairs, eval_neg_cnt,ntype2id,etype2id):
        model.eval()
        t0 = time.time()

        p_h = {}
        with th.no_grad():
            for i, blocks in enumerate(minibatch_blocks):
                mp_h = model.encoder.forward_mb(blocks)
                mini_ntype, mini_idx = minibatch_info[i]
                if p_h.get(mini_ntype, None) is None:
                    p_h[mini_ntype] = th.empty((eval_g.number_of_nodes(mini_ntype), dim_size), device=device)
                p_h[mini_ntype][mini_idx] = mp_h[mini_ntype]

            test_head_ids, test_etypes, test_tail_ids, test_htypes, test_ttypes = pos_pairs
            test_neg_head_ids, _, test_neg_tail_ids, test_neg_htypes, test_neg_ttypes = neg_pairs

            mrr = 0
            mr = 0
            hit1 = 0
            hit3 = 0
            hit10 = 0
            pos_batch_size = 1000
            pos_cnt = test_head_ids.shape[0]
            total_cnt = 0

            # unique test head and tail nodes
            if test_htypes is None:
                unique_neg_head_ids = th.unique(test_neg_head_ids)
                unique_neg_tail_ids = th.unique(test_neg_tail_ids)
                unique_neg_htypes = None
                unique_neg_ttypes = None
            else:
                unique_neg_head_ids = []
                unique_neg_tail_ids = []
                unique_neg_htypes = []
                unique_neg_ttypes = []
                for nt in eval_g.ntypes:
                    cols = (test_neg_htypes == ntype2id[nt])
                    unique_ids = th.unique(test_neg_head_ids[cols])
                    unique_neg_head_ids.append(unique_ids)
                    unique_neg_htypes.append(th.full((unique_ids.shape[0],),  ntype2id[nt]))

                    cols = (test_neg_ttypes ==  ntype2id[nt])
                    unique_ids = th.unique(test_neg_tail_ids[cols])
                    unique_neg_tail_ids.append(unique_ids)
                    unique_neg_ttypes.append(th.full((unique_ids.shape[0],),  ntype2id[nt]))
                unique_neg_head_ids = th.cat(unique_neg_head_ids)
                unique_neg_tail_ids = th.cat(unique_neg_tail_ids)
                unique_neg_htypes = th.cat(unique_neg_htypes)
                unique_neg_ttypes = th.cat(unique_neg_ttypes)

            if eval_neg_cnt > 0:
                total_neg_head_seed = th.randint(unique_neg_head_ids.shape[0],
                                                 (eval_neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
                total_neg_tail_seed = th.randint(unique_neg_tail_ids.shape[0],
                                                 (eval_neg_cnt * ((pos_cnt // pos_batch_size) + 1),))
            for p_i in range(int((pos_cnt + pos_batch_size - 1) // pos_batch_size)):
                print("Eval {}-{}".format(p_i * pos_batch_size,
                                          (p_i + 1) * pos_batch_size \
                                              if (p_i + 1) * pos_batch_size < pos_cnt \
                                              else pos_cnt))
                sub_test_head_ids = test_head_ids[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                sub_test_etypes = test_etypes[p_i * pos_batch_size: \
                                              (p_i + 1) * pos_batch_size \
                                                  if (p_i + 1) * pos_batch_size < pos_cnt \
                                                  else pos_cnt]
                sub_test_tail_ids = test_tail_ids[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]

                if test_htypes is None:
                    phead_emb = p_h['node'][sub_test_head_ids]
                    ptail_emb = p_h['node'][sub_test_tail_ids]
                else:
                    sub_test_htypes = test_htypes[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                    sub_test_ttypes = test_ttypes[p_i * pos_batch_size: \
                                                  (p_i + 1) * pos_batch_size \
                                                      if (p_i + 1) * pos_batch_size < pos_cnt \
                                                      else pos_cnt]
                    phead_emb = th.empty((sub_test_head_ids.shape[0], dim_size), device=device)
                    ptail_emb = th.empty((sub_test_tail_ids.shape[0], dim_size), device=device)
                    for nt in eval_g.ntypes:
                        if nt in p_h:
                            loc = (sub_test_htypes ==  ntype2id[nt])
                            phead_emb[loc] = p_h[nt][sub_test_head_ids[loc]]
                            loc = (sub_test_ttypes ==  ntype2id[nt])
                            ptail_emb[loc] = p_h[nt][sub_test_tail_ids[loc]]

                pos_scores = model.linkPredictor.calc_pos_score_with_rids(phead_emb, ptail_emb, sub_test_etypes,etype2id,device)
                pos_scores = F.logsigmoid(pos_scores).reshape(phead_emb.shape[0], -1).detach().cpu()

                if eval_neg_cnt > 0:
                    neg_head_seed = total_neg_head_seed[p_i * eval_neg_cnt:(p_i + 1) * eval_neg_cnt]
                    neg_tail_seed = total_neg_tail_seed[p_i * eval_neg_cnt:(p_i + 1) * eval_neg_cnt]
                    seed_test_neg_head_ids = unique_neg_head_ids[neg_head_seed]
                    seed_test_neg_tail_ids = unique_neg_tail_ids[neg_tail_seed]
                    if test_neg_htypes is not None:
                        seed_test_neg_htypes = unique_neg_htypes[neg_head_seed]
                        seed_test_neg_ttypes = unique_neg_ttypes[neg_tail_seed]
                else:
                        seed_test_neg_head_ids = unique_neg_head_ids
                        seed_test_neg_tail_ids = unique_neg_tail_ids
                        seed_test_neg_htypes = unique_neg_htypes
                        seed_test_neg_ttypes = unique_neg_ttypes

                neg_batch_size = 10000
                head_neg_cnt = seed_test_neg_head_ids.shape[0]
                tail_neg_cnt = seed_test_neg_tail_ids.shape[0]
                t_neg_score = []
                h_neg_score = []
                for n_i in range(int((head_neg_cnt + neg_batch_size - 1) // neg_batch_size)):
                    sub_test_neg_head_ids = seed_test_neg_head_ids[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < head_neg_cnt
                                                                   else head_neg_cnt]
                    if test_htypes is None:
                        nhead_emb = p_h['node'][sub_test_neg_head_ids]
                    else:
                        sub_test_neg_htypes = seed_test_neg_htypes[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < head_neg_cnt
                                                                   else head_neg_cnt]

                        nhead_emb = th.empty((sub_test_neg_head_ids.shape[0], dim_size), device=device)
                        for nt in eval_g.ntypes:
                            if nt in p_h:
                                loc = (sub_test_neg_htypes ==  ntype2id[nt])
                                nhead_emb[loc] = p_h[nt][sub_test_neg_head_ids[loc]]

                    h_neg_score.append(
                        model.linkPredictor.calc_neg_head_score(nhead_emb,
                                                                 ptail_emb,
                                                                 sub_test_etypes,
                                                                 1,
                                                                 ptail_emb.shape[0],
                                                                 nhead_emb.shape[0],etype2id,device).reshape(-1, nhead_emb.shape[
                        0]).detach().cpu())

                for n_i in range(int((tail_neg_cnt + neg_batch_size - 1) // neg_batch_size)):
                    sub_test_neg_tail_ids = seed_test_neg_tail_ids[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < tail_neg_cnt
                                                                   else tail_neg_cnt]

                    if test_htypes is None:
                        ntail_emb = p_h['node'][sub_test_neg_tail_ids]
                    else:
                        sub_test_neg_ttypes = seed_test_neg_ttypes[n_i * neg_batch_size: \
                                                                   (n_i + 1) * neg_batch_size \
                                                                       if (n_i + 1) * neg_batch_size < tail_neg_cnt
                                                                   else tail_neg_cnt]
                        ntail_emb = th.empty((sub_test_neg_tail_ids.shape[0], dim_size), device=device)
                        for nt in eval_g.ntypes:
                            if nt in p_h:
                                loc = (sub_test_neg_ttypes ==  ntype2id[nt])
                                ntail_emb[loc] = p_h[nt][sub_test_neg_tail_ids[loc]]

                    t_neg_score.append(model.linkPredictor.calc_neg_tail_score(phead_emb,
                                                                 ntail_emb,
                                                                 sub_test_etypes,
                                                                 1,
                                                                 phead_emb.shape[0],
                                                                 ntail_emb.shape[0],etype2id,device).reshape(-1, ntail_emb.shape[
                        0]).detach().cpu())
                t_neg_score = th.cat(t_neg_score, dim=1)
                h_neg_score = th.cat(h_neg_score, dim=1)
                t_neg_score = F.logsigmoid(t_neg_score)
                h_neg_score = F.logsigmoid(h_neg_score)

                canonical_etypes = eval_g.canonical_etypes
                for idx in range(phead_emb.shape[0]):
                    if test_htypes is None:
                        tail_pos = eval_g.has_edges_between(
                            th.full((seed_test_neg_tail_ids.shape[0],), sub_test_head_ids[idx]).long(),
                            seed_test_neg_tail_ids,
                            etype=test_g.etypes[(sub_test_etypes[idx].numpy().item())])
                        head_pos = eval_g.has_edges_between(seed_test_neg_head_ids,
                                                            th.full((seed_test_neg_head_ids.shape[0],),
                                                                    sub_test_tail_ids[idx]).long(),
                                                            etype=test_g.etypes[(sub_test_etypes[idx].numpy().item())])
                        loc = tail_pos == 1
                        t_neg_score[idx][loc] += pos_scores[idx]
                        loc = head_pos == 1
                        h_neg_score[idx][loc] += pos_scores[idx]

                    else:
                        head_type = test_g.ntypes[(sub_test_htypes[idx].numpy())]
                        tail_type = test_g.ntypes[(sub_test_ttypes[idx].numpy())]

                        for t in eval_g.ntypes:
                            if (head_type, test_g.etypes[(sub_test_etypes[idx].numpy().item())], t) in canonical_etypes:
                                loc = (seed_test_neg_ttypes ==  ntype2id[t])
                                t_neg_tail_ids = seed_test_neg_tail_ids[loc]

                                # there is some neg tail in this type
                                if t_neg_tail_ids.shape[0] > 0:
                                    tail_pos = eval_g.has_edges_between(
                                        th.full((t_neg_tail_ids.shape[0],), sub_test_head_ids[idx]).long(),
                                        t_neg_tail_ids,
                                        etype=(head_type,
                                               test_g.etypes[(sub_test_etypes[idx].numpy().item())],
                                               t))
                                    t_neg_score[idx][loc][tail_pos == 1] += pos_scores[idx]
                            if (t, test_g.etypes[(sub_test_etypes[idx].numpy().item())], tail_type) in canonical_etypes:
                                loc = (seed_test_neg_htypes == ntype2id[t])
                                t_neg_head_ids = seed_test_neg_head_ids[loc]

                                # there is some neg head in this type
                                if t_neg_head_ids.shape[0] > 0:
                                    head_pos = eval_g.has_edges_between(t_neg_head_ids,
                                                                        th.full((t_neg_head_ids.shape[0],),
                                                                                sub_test_tail_ids[idx]).long(),
                                                                        etype=(t,
                                                                               test_g.etypes[(sub_test_etypes[idx].numpy().item())]
                                                                               ,
                                                                               tail_type))
                                    h_neg_score[idx][loc][head_pos == 1] += pos_scores[idx]
                neg_score = th.cat([h_neg_score, t_neg_score], dim=1)

                rankings = th.sum(neg_score >= pos_scores, dim=1) + 1
                rankings = rankings.cpu().detach().numpy()
                for ranking in rankings:
                    mrr += 1.0 / ranking
                    mr += float(ranking)
                    hit1 += 1.0 if ranking <= 1 else 0.0
                    hit3 += 1.0 if ranking <= 3 else 0.0
                    hit10 += 1.0 if ranking <= 10 else 0.0
                    total_cnt += 1
        res="MRR {}\nMR {}\nHITS@1 {}\nHITS@3 {}\nHITS@10 {}".format(mrr / total_cnt,
                                                                       mr / total_cnt,
                                                                       hit1 / total_cnt,
                                                                       hit3 / total_cnt,
                                                                       hit10 / total_cnt)
        print("MRR {}\nMR {}\nHITS@1 {}\nHITS@3 {}\nHITS@10 {}".format(mrr / total_cnt,
                                                                       mr / total_cnt,
                                                                       hit1 / total_cnt,
                                                                       hit3 / total_cnt,
                                                                       hit10 / total_cnt))
        t1 = time.time()
        print("Full eval {} exmpales takes {} seconds".format(pos_scores.shape[0], t1 - t0))
        return res

    ntype2id = {}
    for i, ntype in enumerate(test_g.ntypes):
            ntype2id[ntype] = i
    etype2id = {}
    for i, etype in enumerate(test_g.etypes):
            etype2id[etype] = i
    train_data=transform_triplets(train_edges, etype2id, ntype2id)
    valid_data = transform_triplets(valid_edges, etype2id, ntype2id)
    test_data = transform_triplets(test_edges, etype2id, ntype2id)
    pos_pairs, neg_pairs=prepare_triplets(train_data, valid_data, test_data)
    minibatch_info, minibatch_blocks=creat_eval_minibatch(test_g, n_layers)
    res=fullgraph_eval(test_g, model, device, dim_size, minibatch_blocks, minibatch_info,
                   pos_pairs, neg_pairs, eval_neg_cnt,ntype2id,etype2id)
    return res
def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.srcnodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb
def evaluate_prfin_for_node_classification(model, seeds, blocks, device, labels, category, use_cuda,loss_func, multilabel=False):
    model.eval()
    #emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        lbl = lbl.cuda()
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
    logits = model.classifier_forward_mb(blocks)[category]
    if multilabel:
        loss = loss_func(logits.squeeze(1),
                         lbl)
    else:
        loss = loss_func(logits.squeeze(1), torch.max(lbl, 1)[1])

    acc = torch.sum(logits.argmax(dim=1) == lbl.argmax(dim=1) ).item() / len(seeds)
    pred = torch.sigmoid(logits).detach().cpu().numpy()
    try:
        acc_auc = roc_auc_score(lbl.cpu().numpy(), pred)
    except ValueError:
        acc_auc = -1
        pass
    return loss, acc,acc_auc
def direct_eval_lppr_link_prediction(test_g, model, train_edges, valid_edges, test_edges, n_hidden,n_layers, eval_neg_cnt=100,use_cuda=True):
    # evaluate PanRep LP module for link prediction
    if use_cuda:
        model.cpu()
        test_g = test_g.to(torch.device("cpu"))
    pr_mrr = "PanRep LP "
    pr_mrr += evaluation_link_prediction(test_g, model, train_edges, valid_edges, test_edges, dim_size=n_hidden,
                                         eval_neg_cnt=eval_neg_cnt,
                                         n_layers=n_layers,
                                         device=torch.device("cpu"))
    if use_cuda:
        model.cuda()

    return pr_mrr
def direct_eval_pr_link_prediction(train_g,test_g,train_edges, valid_edges, test_edges,fanout,batch_size,n_hidden,ntype2id,ng_rate,l2norm,
                                   n_layers,n_lp_epochs,embeddings,use_cuda,device):
    sampler = InfomaxNodeRecNeighborSampler(train_g, [fanout] * (n_layers), device=device)
    pr_train_ind=list(sampler.hetero_map.keys())
    lp_sampler = LinkPredictorEvalSampler(train_g, [fanout] * (1),device=device)
    lp_loader = DataLoader(dataset=pr_train_ind,
                        batch_size=batch_size,
                        collate_fn=lp_sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)
    lp_model=DLinkPredictorOnlyRel(out_dim=n_hidden,etypes=train_g.etypes,ntype2id=ntype2id,edg_pct=1,ng_rate=ng_rate,use_cuda=True)
    if use_cuda:
        lp_model.cuda()
    lp_optimizer = torch.optim.Adam(lp_model.parameters(), lr=5e-2, weight_decay=l2norm)
    for epoch in range(n_lp_epochs):
        lp_model.train()

        lp_optimizer.zero_grad()
        for i, (seeds, blocks) in enumerate(lp_loader):
            embs={}
            for ntype in seeds:
                embs[ntype]=embeddings[ntype][seeds[ntype]].to(device)
            loss= lp_model.forward_mb(g=blocks[0],embed=embs)

            loss.backward()
            lp_optimizer.step()
            print("Link Predict finetune loss: {:.4f} Epoch {:05d} | Batch {:03d}".format(loss.item(), epoch, i))
    if use_cuda:
        lp_model.cpu()
        train_g = train_g.to(torch.device("cpu"))
        test_g=test_g.to(torch.device("cpu"))
    pr_mrr=evaluation_link_prediction_wembeds(test_g, lp_model,embeddings, train_edges, valid_edges, test_edges, dim_size=n_hidden,
                                      eval_neg_cnt=100,
                                      n_layers=n_layers,
                                      device=torch.device("cpu"))
    if use_cuda:
        train_g = train_g.to(device)
    return pr_mrr

def finetune_pr_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, batch_size,
                                       n_hidden, ntype2id, ng_rate,fanout, l2norm,
                                       n_layers, n_lp_fintune_epochs,lr_lp_ft, use_cuda, device,learn_rel_embed=False):
    if learn_rel_embed:
        finetune_link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=train_g.etypes,
                                                     ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1, ng_rate=ng_rate)
    else:
        finetune_link_predictor = LinkPredictor(out_dim=n_hidden, etypes=train_g.etypes,
                                       ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                       ng_rate=ng_rate)

    finetune_lp_model= copy.deepcopy(model)
    finetune_lp_model.linkPredictor=finetune_link_predictor

    adjust_pr_lr = True
    if adjust_pr_lr:
        cl_params = set(list(finetune_lp_model.linkPredictor.parameters()))
        tot_params = set(list(model.parameters()))
        res_params = list(tot_params.difference(cl_params))
        finetune_lp_optimizer = torch.optim.Adam([{'params': res_params},
                                      {'params': finetune_lp_model.linkPredictor.parameters(), 'lr': lr_lp_ft}],
                                     lr=lr_lp_ft / 10, weight_decay=l2norm)
    else:
        finetune_lp_optimizer = torch.optim.Adam(model.parameters(), lr=lr_lp_ft, weight_decay=l2norm)

    sampler = InfomaxNodeRecNeighborSampler(train_g, [fanout] * (n_layers), device=device)
    pr_train_ind = list(sampler.hetero_map.keys())
    loader = DataLoader(dataset=pr_train_ind,
                        batch_size=batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)
    if use_cuda:
        finetune_lp_model.cuda()


    for epoch in range(n_lp_fintune_epochs):
        finetune_lp_model.train()

        finetune_lp_optimizer.zero_grad()
        for i, (seeds, blocks) in enumerate(loader):
            loss = finetune_lp_model.link_predictor_forward_mb(p_blocks=blocks)
            loss.backward()
            finetune_lp_optimizer.step()

            print("Train Loss: {:.4f} Epoch {:05d} | Batch {:03d}".format(loss.item(), epoch, i))

    if use_cuda:
        finetune_lp_model.cpu()
        test_g=test_g.to(torch.device("cpu"))

    pr_mrr = "PanRep LPFT "
    pr_mrr += evaluation_link_prediction(test_g, finetune_lp_model, train_edges, valid_edges, test_edges, dim_size=n_hidden,
                                         eval_neg_cnt=100,
                                         n_layers=n_layers,
                                         device=torch.device("cpu"))

    if use_cuda:
        test_g = test_g.to(device)
    return pr_mrr

def eval_panrep(model,dataloader):
    model.eval()

    print("==========Start evaluating===============")
    for i, (seeds, blocks) in enumerate(dataloader):

        loss, embeddings = model.forward_mb(p_blocks=blocks)
    print("=============Evaluation finished=============")
    return

def _fit(args):
    #dblp add the original node split and test...

    rw_supervision = args.rw_supervision
    n_layers = args.n_layers
    use_reconstruction_loss = args.use_reconstruction_loss
    num_cluster = args.num_cluster
    use_node_motif = args.use_node_motif
    k_fold = args.k_fold
    test_edge_split = args.test_edge_split
    use_link_prediction = args.use_link_prediction
    motif_cluster = args.motif_cluster
    use_infomax_loss = args.use_infomax_loss
    mask_links = args.mask_links
    n_hidden = args.n_hidden
    n_bases = args.n_bases
    dropout = args.dropout
    fanout = args.fanout
    use_self_loop = args.use_self_loop
    ng_rate = args.ng_rate
    n_epochs = args.n_epochs
    lr = args.lr
    n_fine_tune_epochs = args.n_fine_tune_epochs
    only_ssl = args.only_ssl
    single_layer = args.single_layer
    if rw_supervision and n_layers==1:
        return ""
    if num_cluster>0:
        args.use_cluster=True
        args.num_clusters=num_cluster
    else:
        args.use_cluster = False
    if args.dataset=="query_biodata":
        if use_reconstruction_loss:
            return ""
    args.k_fold=k_fold
    args.few_shot=False
    args.test_edge_split=test_edge_split
    args.use_link_prediction=use_link_prediction
    args.rw_supervision = rw_supervision
    args.motif_clusters=motif_cluster
    args.use_node_motifs=use_node_motif
    train_idx, test_idx, val_idx, labels, category, num_classes, masked_node_types, metapaths, \
    train_edges, test_edges, valid_edges, train_g, valid_g, test_g=\
        load_univ_hetero_data(args)
    if args.focus:
        args.focus_category=category
    else:
        args.focus_category=None

    # sampler parameters
    batch_size = 16 *1024
    l2norm=0.0001

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)


    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    # create model
    #dgl.contrib.sampling.sampler.EdgeSampler(g['customer_to_transaction'], batch_size=100)
    use_reconstruction_loss = use_reconstruction_loss
    use_infomax_loss = use_infomax_loss

    # not used currently
    num_masked_nodes = -1
    node_masking = False
    loss_over_all_nodes = True
    pct_masked_edges = -1
    negative_rate = -1
    link_prediction = use_link_prediction
    mask_links = mask_links


    # for the embedding layer
    out_size_dict={}
    for name in train_g.ntypes:
        if name not in masked_node_types:
            if num_cluster>0:
                out_size_dict[name]=train_g.nodes[name].data['h_clusters'].size(1)
            else:
                out_size_dict[name] = train_g.nodes[name].data['h_f'].size(1)
        else:
            out_size_dict[name] =0

    # for the motif layer

    out_motif_dict = {}
    if use_node_motif:
        for name in train_g.ntypes:
            out_motif_dict[name] = train_g.nodes[name].data['motifs'].size(1)

    ntype2id = {}
    for i, ntype in enumerate(train_g.ntypes):
            ntype2id[ntype] = i
    if args.encoder=='RGCN':
        encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=train_g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=train_g.etypes,
                                        ntypes=train_g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)
    elif args.encoder=='RGAT':
        encoder = EncoderRelGraphAttentionHetero(
                                            n_hidden,
                                            etypes=train_g.etypes,
                                            ntypes=train_g.ntypes,
                                            num_hidden_layers=n_layers,
                                            dropout=dropout,
                                            use_self_loop=use_self_loop)


    if rw_supervision:
        mrw_interact = {}
        for ntype in train_g.ntypes:
            mrw_interact[ntype]=[]
            for neighbor_ntype in train_g.ntypes:
                mrw_interact[ntype] +=[neighbor_ntype]

        metapathRWSupervision = MetapathRWalkerSupervision(in_dim=n_hidden, negative_rate=10,
                                                           device=device,mrw_interact=mrw_interact)
    else:
        metapathRWSupervision=None

    link_predictor = None
    learn_rel_embed=False
    if use_link_prediction:
        if learn_rel_embed:
            link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=train_g.etypes,
                                      ntype2id=ntype2id, use_cuda=use_cuda,edg_pct=1,ng_rate=ng_rate)
        else:
            link_predictor = LinkPredictor(out_dim=n_hidden, etypes=train_g.etypes,
                                                         ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                                         ng_rate=ng_rate)

    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=train_g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict=out_size_dict,
                             masked_node_types=masked_node_types,
                             loss_over_all_nodes=loss_over_all_nodes,
                             use_infomax_task=use_infomax_loss,
                             use_reconstruction_task=use_reconstruction_loss,
                             use_node_motif=use_node_motif,
                             link_predictor=link_predictor,
                             out_motif_dict=out_motif_dict,
                             use_cluster=num_cluster>0,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision, focus_category=args.focus_category)


    if use_cuda:
        model.cuda()


    if labels is not None and len(labels.shape)>1:
        if use_cuda:
            labels = labels.cuda()
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))
        #TODO val set not used currently


    if use_cuda:
        train_g=train_g.to(device)

    val_pct = 0.1
    evaluate_every = 50
    sampler = InfomaxNodeRecNeighborSampler(train_g, [fanout] * (n_layers), device=device)
    evaluate_panrep = False
    num_workers=0
    if evaluate_panrep:
        pr_node_ids=list(sampler.hetero_map.keys())
        pr_val_ind=list(np.random.choice(len(pr_node_ids), int(len(pr_node_ids)*val_pct), replace=False))
        pr_train_ind=list(set(list(np.arange(len(pr_node_ids)))).difference(set(pr_val_ind)))
        loader = DataLoader(dataset=pr_train_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=num_workers)


        valid_loader = DataLoader(dataset=pr_val_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=num_workers)
    else:
        pr_train_ind=list(sampler.hetero_map.keys())
        loader = DataLoader(dataset=pr_train_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers= num_workers)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    # training loop
    print("start training...")


    for epoch in range(n_epochs):
        model.train()

        optimizer.zero_grad()
        for i, (seeds, blocks) in enumerate(loader):
            if rw_supervision:
                st=time.time()
                rw_neighbors=generate_rwalks(g=train_g,metapaths=metapaths,samples_per_node=4,device=device)
                print('Sampling rw time: '+str(time.time()-st))
            else:
                rw_neighbors=None
            loss, embeddings = model.forward_mb(p_blocks=blocks,rw_neighbors=rw_neighbors)
            loss.backward()
            optimizer.step()


            print("Train Loss: {:.4f} Epoch {:05d} | Batch {:03d}".format(loss.item(), epoch, i))
        if evaluate_panrep and epoch % evaluate_every == 0:
            eval_panrep(model=model, dataloader=valid_loader)


    ## Evaluate before finetune
    model.eval()
    if use_cuda:
        model.cpu()
        model.encoder.cpu()
        train_g=train_g.to(torch.device("cpu"))
    with torch.no_grad():
        embeddings = model.encoder.forward(train_g)
    if use_cuda:
        model.cuda()
        model.encoder.cuda()
        train_g=train_g.to(device)
    #calculate entropy
    entropy = calculate_entropy(embeddings)
    print("Entropy: "+str(entropy))
    # evaluate link prediction
    pr_mrr="PanRep "
    eval_lp=args.test_edge_split>0
    if eval_lp:
        # Evaluate PanRep embeddings for link prediction
        n_lp_epochs=n_epochs
        #pr_mrr+=direct_eval_pr_link_prediction(train_g, test_g,train_edges, valid_edges, test_edges,
        #                               fanout, batch_size, n_hidden, ntype2id, ng_rate, l2norm,
        #                               n_layers, n_lp_epochs, embeddings, use_cuda, device)


        if use_link_prediction:
            # Evaluated LP model in PanRep for link prediction
            pr_mrr+=direct_eval_lppr_link_prediction(test_g, model, train_edges, valid_edges, test_edges, n_hidden, n_layers,
                                              eval_neg_cnt=100, use_cuda=True)

        # finetune PanRep for link prediction
        n_lp_fintune_epochs=n_epochs
        lr_lp_ft=lr
        pr_mrr+=finetune_pr_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, batch_size,
                                        n_hidden, ntype2id, ng_rate, fanout, l2norm,
                                        n_layers, n_lp_fintune_epochs, lr_lp_ft, use_cuda, device,learn_rel_embed=learn_rel_embed)


    eval_nc=False
    if eval_nc and labels is not None:
        multilabel = True
        feats = embeddings[category]
        labels_i=np.argmax(labels.cpu().numpy(),axis=1)
        lr_d=lr
        n_cepochs=1000#n_epochs
        test_acc_prembed = mlp_classifier(
            feats, use_cuda, n_hidden, lr_d, n_cepochs, multilabel, num_classes, labels, train_idx, val_idx, test_idx, device)
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,macro_str,micro_str = evaluate_results_nc(
            feats[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)

        ## Finetune PanRep
        # add the target category in the sampler
        sampler = InfomaxNodeRecNeighborSampler(train_g, [fanout] * (n_layers), device=device,category=category)
        fine_tune_loader = DataLoader(dataset=list(train_idx),
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers= num_workers)


        # validation sampler
        val_sampler = HeteroNeighborSampler(train_g, category, [fanout] * n_layers, True)
        _, val_blocks = val_sampler.sample_blocks(val_idx)

        # test sampler
        test_sampler = HeteroNeighborSampler(train_g, category, [fanout] * n_layers, True)
        _, test_blocks = test_sampler.sample_blocks(test_idx)
        # set up fine_tune epochs
        n_init_fine_epochs=0#int(n_fine_tune_epochs/2)
        n_fine_tune_epochs=n_fine_tune_epochs-n_init_fine_epochs


        # donwstream classifier

        model.classifier=ClassifierMLP(input_size=n_hidden,hidden_size=n_hidden,out_size=num_classes,single_layer=single_layer)
        if multilabel is False:
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            loss_func = torch.nn.BCEWithLogitsLoss()
        if use_cuda:
            model.cuda()
            feats=feats.cuda()
        labels=labels#.float()
        best_test_acc=0
        best_val_acc=0
        #NodeClassifierRGCN(in_dim= n_hidden,out_dim=num_classes, rel_names=g.etypes,num_bases=n_bases)
        optimizer_init = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=l2norm)
        lbl = labels
        for epoch in range(n_init_fine_epochs):
                model.classifier.train()
                optimizer.zero_grad()
                t0 = time.time()

                logits = model.classifier.forward(feats)
                if multilabel:
                    loss = loss_func(logits[train_idx].squeeze(1),
                                                              lbl[train_idx])
                else:
                    loss=loss_func(logits[train_idx].squeeze(1),torch.max(lbl[train_idx], 1)[1] )
                loss.backward()
                optimizer_init.step()

                pred = torch.sigmoid(logits).detach().cpu().numpy()
                train_acc = roc_auc_score(labels.cpu()[train_idx].numpy(),
                                          pred[train_idx], average='macro')
                val_acc = roc_auc_score(labels.cpu()[val_idx].numpy(),
                                        pred[val_idx], average='macro')
                test_acc = roc_auc_score(labels.cpu()[test_idx].numpy()
                                         , pred[test_idx], average='macro')
                test_acc_w = roc_auc_score(labels.cpu()[test_idx].numpy()
                                           , pred[test_idx], average='weighted')

                macro_test,micro_test=macro_micro_f1(
                    torch.max(labels[test_idx], 1)[1].cpu(), torch.max(logits[test_idx], 1)[1].cpu())

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc

                if epoch % 5 == 0:
                    print(
                        'Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Weighted Test Acc %.4f' % (
                            loss.item(),
                            train_acc.item(),
                            val_acc.item(),
                            best_val_acc.item(),
                            test_acc.item(),
                            best_test_acc.item(), test_acc_w.item()
                        ))
        print()

        adjust_pr_lr=False
        if adjust_pr_lr:
            cl_params = set(list(model.classifier.parameters()))
            tot_params = set(list(model.parameters()))
            res_params = list(tot_params.difference(cl_params))
            optimizer = torch.optim.Adam([{'params':res_params},
                                          {'params':model.classifier.parameters(), 'lr': lr}],
                                         lr=lr/10, weight_decay=l2norm)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
        # training loop
        print("start training...")
        forward_time = []
        backward_time = []
        model.train()
        dur = []

        # cancel the link preidction and rw superivsion tasks since the batch graph sampled will not have
        # connections among different nodes. Consider full graph sampling but evaluate only in the category node.
        model.link_prediction_task=False
        model.rw_supervision_task=False


        for epoch in range(n_fine_tune_epochs):
            model.train()
            optimizer.zero_grad()
            if epoch > 3:
                t0 = time.time()

            for i, (seeds, blocks) in enumerate(fine_tune_loader):
                batch_tic = time.time()
                # need to copy the features
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                #emb = extract_embed(node_embed, blocks[0])
                lbl = labels[seeds[category]]
                logits = model.classifier_forward_mb(blocks)[category]
                if multilabel:
                    log_loss = loss_func(logits.squeeze(1),
                                     lbl)
                else:
                    log_loss = loss_func(logits.squeeze(1), torch.max(lbl, 1)[1])
                print("Log loss :"+str(log_loss.item()))
                if only_ssl:
                    loss=log_loss
                else:
                    if rw_supervision:
                        st = time.time()
                        rw_neighbors = generate_rwalks(g=train_g, metapaths=metapaths, samples_per_node=4,
                                                       device=device)
                        print('Sampling rw time: ' + str(time.time() - st))
                    else:
                        rw_neighbors = None
                    pr_loss, embeddings = model.forward_mb(p_blocks=blocks, rw_neighbors=rw_neighbors)
                    loss=pr_loss+log_loss
                loss.backward()
                optimizer.step()

                train_acc = torch.sum(logits.argmax(dim=1) == lbl.argmax(dim=1)).item() / len(seeds[category])
                pred = torch.sigmoid(logits).detach().cpu().numpy()
                try:
                    train_acc_auc = roc_auc_score(lbl.cpu().numpy(), pred)
                except ValueError:
                    train_acc_auc=-1
                print(
                    "Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} |Train Acc AUC: {:.4f}| Train Loss: {:.4f} | Time: {:.4f}".
                    format(epoch, i, train_acc, train_acc_auc, loss.item(), time.time() - batch_tic))

            if epoch > 3:
                dur.append(time.time() - t0)

            val_loss, val_acc,val_acc_auc = evaluate_prfin_for_node_classification\
                (model, val_idx, val_blocks, device, labels, category, use_cuda,loss_func, multilabel=multilabel)
            print("Epoch {:05d} | Valid Acc: {:.4f} |Valid Acc Auc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
                  format(epoch, val_acc, val_acc_auc,val_loss.item(), np.average(dur)))
        print()

        # full graph evaluation here
        model.eval()
        if use_cuda:
            model.cpu()
            model.encoder.cpu()
            train_g=train_g.to(torch.device("cpu"))
        with torch.no_grad():
            embeddings = model.encoder.forward(train_g)
            logits=model.classifier_forward(train_g)[category]

        acc = torch.sum(logits[test_idx].argmax(dim=1) == labels[test_idx].cpu().argmax(dim=1)).item() / len(test_idx)
        print("Test accuracy: {:4f}".format(acc))
        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

        feats = embeddings[category]
        test_acc_ftembed=mlp_classifier(
            feats, use_cuda, n_hidden, lr_d, n_cepochs, multilabel, num_classes, labels, train_idx, val_idx, test_idx, device)
        labels_i=np.argmax(labels.cpu().numpy(),axis=1)
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,finmacro_str,finmicro_str = evaluate_results_nc(
            feats[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
        print("With logits")
        svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std, macro_str_log, micro_str_log = evaluate_results_nc(
            logits[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
        return " | PanRep " +macro_str+" "+micro_str+ " | Test accuracy: {:4f} | ".format(acc)+" " \
        "| Finetune "+finmacro_str+" "+finmicro_str+"PR MRR : "+pr_mrr+" Logits: "+macro_str_log+" "+\
               micro_str_log +" Entropy "+ str(entropy) + " Test acc PR embed "+str(test_acc_prembed)+\
               " Test acc PRft embed "+str(test_acc_ftembed)
    else:
        return"PR MRR : " + pr_mrr + " Entropy " + str(entropy)
def macro_micro_f1(y_test, y_pred):
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print("Macro micro f1 " +str(macro_f1)+ " "+str(micro_f1))
    return macro_f1, micro_f1


def fit(args):
        '''
        TODO
            best results 700 hidden units so far
        '''
        args.splitpct = 0.1
        n_epochs_list = [800]
        n_hidden_list = [250]
        n_layers_list = []
        n_fine_tune_epochs_list= [50]#[20,50]#[30,50,150]
        n_bases_list = [2]
        lr_list = [2e-3]
        dropout_list = [0.1]
        focus=False
        fanout=10
        test_edge_split_list = [0.05
                                ]
        use_link_prediction_list = [True]
        use_reconstruction_loss_list =[True]#[False,True]
        use_infomax_loss_list = [True]#[False,True]
        use_node_motif_list = [False]
        rw_supervision_list=[False]
        num_cluster_list=[6]
        K_list=[0]#[2,5,10,15]
        motif_cluster_list=[5]#[2,6,10]
        #mask_links_list = [False]
        use_self_loop_list=[True]
        ng_rate_list=[5]
        single_layer_list=[True]
        only_ssl_list=[True]
        results={}
        for n_epochs in n_epochs_list:
            for n_fine_tune_epochs in n_fine_tune_epochs_list:
                for n_hidden in n_hidden_list:
                    for n_layers in n_layers_list:
                        for n_bases in n_bases_list:
                            for test_edge_split in test_edge_split_list:
                                for lr in lr_list:
                                    for dropout in dropout_list:
                                        for use_infomax_loss in use_infomax_loss_list:
                                            for use_link_prediction in use_link_prediction_list:
                                                for use_reconstruction_loss in use_reconstruction_loss_list:
                                                    #for mask_links in mask_links_list:
                                                        for use_self_loop in use_self_loop_list:
                                                            for use_node_motif in use_node_motif_list:
                                                                for num_cluster in num_cluster_list:
                                                                    for single_layer in single_layer_list:
                                                                        for motif_cluster in motif_cluster_list:
                                                                            for rw_supervision in rw_supervision_list:
                                                                                for k_fold in K_list:
                                                                                    for ng_rate in ng_rate_list:
                                                                                        for only_ssl in only_ssl_list:
                                                                                            if not use_reconstruction_loss and not \
                                                                                                    use_infomax_loss and not use_link_prediction\
                                                                                                    and not use_node_motif and not rw_supervision:
                                                                                                continue
                                                                                            else:

                                                                                                mask_links=False
                                                                                                args.focus=focus
                                                                                                args.rw_supervision = rw_supervision
                                                                                                args.n_layers = n_layers
                                                                                                args.use_reconstruction_loss = use_reconstruction_loss
                                                                                                args.num_cluster = num_cluster
                                                                                                args.use_node_motif = use_node_motif
                                                                                                args.k_fold = k_fold
                                                                                                args.test_edge_split = test_edge_split
                                                                                                args.use_link_prediction = use_link_prediction
                                                                                                args.motif_cluster = motif_cluster
                                                                                                args.use_infomax_loss = use_infomax_loss
                                                                                                args.mask_links = mask_links
                                                                                                args.n_hidden = n_hidden
                                                                                                args.n_bases = n_bases
                                                                                                args.dropout = dropout
                                                                                                args.fanout = fanout
                                                                                                args.use_self_loop = use_self_loop
                                                                                                args.ng_rate = ng_rate
                                                                                                args.n_epochs = n_epochs
                                                                                                args.lr = lr
                                                                                                args.n_fine_tune_epochs = n_fine_tune_epochs
                                                                                                args.only_ssl = only_ssl
                                                                                                args.single_layer = single_layer
                                                                                                acc = _fit(args)
                                                                                                results[(n_epochs,
                                                                                                         n_fine_tune_epochs,
                                                                                                         n_layers,
                                                                                                         n_hidden,
                                                                                                         n_bases,
                                                                                                         fanout, lr,
                                                                                                         dropout,
                                                                                                         use_link_prediction,
                                                                                                         use_reconstruction_loss,
                                                                                                         use_infomax_loss,
                                                                                                         mask_links,
                                                                                                         use_self_loop,
                                                                                                         use_node_motif,
                                                                                                         num_cluster,
                                                                                                         single_layer,
                                                                                                         motif_cluster,
                                                                                                         k_fold,
                                                                                                         rw_supervision,
                                                                                                         ng_rate,
                                                                                                         only_ssl,
                                                                                                         focus,
                                                                                                         test_edge_split)] = acc
                                                                                                print(args)
                                                                                                result = " acc {}".format(
                                                                                                    acc)
                                                                                                print(result)
        results[str(args)]=1
        file=args.dataset+'-'+str(datetime.date(datetime.now()))+"-"+str(datetime.time(datetime.now()))
        pickle.dump(results, open(os.path.join("results/universal_task/", file + ".pickle"), "wb"),
                    protocol=4)
        return

def kmeans_test(X, y, n_clusters, repeat=10):
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)


def svm_test(X, y, test_sizes=(0.2, 0.4, 0.6, 0.8), repeat=10):
    random_states = [182318 + i for i in range(repeat)]
    result_macro_f1_list = []
    result_micro_f1_list = []
    for test_size in test_sizes:
        macro_f1_list = []
        micro_f1_list = []
        for i in range(repeat):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, random_state=random_states[i])
            svm = LinearSVC(dual=False)
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            macro_f1_list.append(macro_f1)
            micro_f1_list.append(micro_f1)
        result_macro_f1_list.append((np.mean(macro_f1_list), np.std(macro_f1_list)))
        result_micro_f1_list.append((np.mean(micro_f1_list), np.std(micro_f1_list)))
    return result_macro_f1_list, result_micro_f1_list


def evaluate_results_nc(embeddings, labels, num_classes):
    print('SVM test')
    svm_macro_f1_list, svm_micro_f1_list = svm_test(embeddings, labels)
    macro_str='Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(macro_f1_mean, macro_f1_std, train_size) for
                                    (macro_f1_mean, macro_f1_std), train_size in
                                    zip(svm_macro_f1_list, [0.8, 0.6, 0.4, 0.2])])

    micro_str='Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(micro_f1_mean, micro_f1_std, train_size) for
                                    (micro_f1_mean, micro_f1_std), train_size in
                                    zip(svm_micro_f1_list, [0.8, 0.6, 0.4, 0.2])])
    print(macro_str)
    print(micro_str)
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,macro_str,micro_str


def mlp_classifier(feats,use_cuda,n_hidden,lr_d,n_cepochs,multilabel,num_classes,labels,train_idx,val_idx,test_idx,device):
    ###
    # Use the encoded features for classification
    # Here we initialize the features using the reconstructed ones

    # feats = g.nodes[category].data['features']
    l2norm = 0.0001
    inp_dim = feats.shape[1]
    model = ClassifierMLP(input_size=inp_dim, hidden_size=n_hidden,out_size=num_classes)

    if use_cuda:
        model.cuda()
        feats=feats.cuda()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_d, weight_decay=l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []
    model.train()

    # TODO find all zero indices rows and remove.
    if len(labels.shape)>1:
        zero_rows=np.where(~(labels).cpu().numpy().any(axis=1))[0]

        train_idx=np.array(list(set(train_idx).difference(set(zero_rows))))
        val_idx = np.array(list(set(val_idx).difference(set(zero_rows))))
        test_idx = np.array(list(set(test_idx).difference(set(zero_rows))))

    train_indices = torch.tensor(train_idx).to(device).long()
    valid_indices = torch.tensor(val_idx).to(device).long()
    test_indices = torch.tensor(test_idx).to(device).long()




    best_val_acc = 0
    best_test_acc = 0
    labels_n=labels
    if multilabel is False:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        loss_func = torch.nn.BCEWithLogitsLoss()

    for epoch in range(n_cepochs):
        optimizer.zero_grad()
        logits = model(feats)
        if multilabel:
            loss = loss_func(logits[train_idx].squeeze(1),
                             labels_n[train_idx])
        else:
            loss = loss_func(logits[train_idx].squeeze(1), torch.max(labels_n[train_idx], 1)[1])
        loss.backward()
        optimizer.step()
        pred = torch.sigmoid(logits).detach().cpu().numpy()

        train_acc = roc_auc_score(labels_n.cpu()[train_indices.cpu()].numpy(),
                                  pred[train_indices.cpu()],average='macro')
        val_acc = roc_auc_score(labels_n.cpu()[valid_indices.cpu()].numpy(),
                                pred[valid_indices.cpu()],average='macro')
        test_acc = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()],average='macro')
        test_acc_w = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                 , pred[test_indices.cpu()], average='weighted')
        macro_test, micro_test = macro_micro_f1(
            torch.max(labels[test_indices], 1)[1].cpu(), torch.max(logits[test_indices], 1)[1].cpu())
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Weighted Test Acc %.4f' % (
                loss.item(),
                train_acc.item(),
                val_acc.item(),
                best_val_acc.item(),
                test_acc.item(),
                best_test_acc.item(),test_acc_w.item()
            ))
    print()
    return best_test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep-FineTune')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=60,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=5,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--lr-d", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
            help="number of training epochs for decoder")
    parser.add_argument("-ec", "--n-cepochs", type=int, default=2,
                        help="number of training epochs for classification")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=100,
                        help="number of masked nodes")
    parser.add_argument("-pct_masked_links", "--pct-masked-links", type=int, default=0.5,
                        help="number of masked links")
    parser.add_argument("-negative_rate", "--negative-rate", type=int, default=4,
                        help="number of negative examples per masked link")


    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("-en", "--encoder", type=str, required=True,
                        help="Encoder to use")
    parser.add_argument("--l2norm", type=float, default=0.0001,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--use-infomax-loss", default=True, action='store_true',
                        help="use infomax task supervision")
    parser.add_argument("--use-reconstruction-loss", default=False, action='store_true',
                        help="use feature reconstruction task supervision")
    parser.add_argument("--node-masking", default=False, action='store_true',
                        help="mask a subset of node features")
    parser.add_argument("--loss-over-all-nodes", default=True, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--link-prediction", default=False, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--mask-links", default=True, action='store_true',
                       help="mask the links to be predicted")
    parser.add_argument("--use-node-motifs", default=True, action='store_true',
                       help="use the node motifs")
    parser.add_argument("--batch-size", type=int, default=5000,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--fanout", type=int, default=10,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--split", type=int, default=0,
                        help="split type.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'oag_na', '--encoder', 'RGCN'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    fit(args)

    '''
    # perform edge neighborhood sampling to generate training graph and data
    g, node_id, edge_type, node_norm, data, labels = \
        utils.generate_sampled_graph_and_labels(
            train_data, args.graph_batch_size, args.graph_split_size,
            num_rels, adj_list, degrees, args.negative_sample,
            args.edge_sampler)
    print("Done edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
        data, labels = data.cuda(), labels.cuda()
    '''