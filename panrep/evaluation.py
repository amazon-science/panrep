'''
This file contains functions to evaluate the link prediction and node classification tasks
'''

import time

import dgl
import numpy as np
import torch
import torch as th
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.nn import functional as F
from torch.utils.data import DataLoader

from classifiers import DLinkPredictorOnlyRel, ClassifierMLP
from node_sampling_masking import InfomaxNodeRecNeighborSampler, LinkPredictorEvalSampler


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
    pr_mrr= evaluation_link_prediction_wembeds(test_g, lp_model, embeddings, train_edges, valid_edges, test_edges, dim_size=n_hidden,
                                               eval_neg_cnt=100,
                                               n_layers=n_layers,
                                               device=torch.device("cpu"))
    if use_cuda:
        train_g = train_g.to(device)
    return pr_mrr


def macro_micro_f1(y_test, y_pred):
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    print("Macro micro f1 " +str(macro_f1)+ " "+str(micro_f1))
    return macro_f1, micro_f1


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

        try:
                train_auc_acc = roc_auc_score(labels_n.cpu()[train_indices.cpu()].numpy(),
                                          pred[train_indices.cpu()], average='macro')
        except ValueError:
                train_auc_acc=0
                pass
        try:
                val_auc_acc = roc_auc_score(labels_n.cpu()[valid_indices.cpu()].numpy(),
                                        pred[valid_indices.cpu()], average='macro')
        except ValueError:
                val_auc_acc=0
                pass
        try:
                test_auc_acc = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                         , pred[test_indices.cpu()], average='macro')
        except ValueError:
                test_auc_acc=0
                pass
        try:
                test_auc_acc_w = roc_auc_score(labels_n.cpu()[test_indices.cpu()].numpy()
                                           , pred[test_indices.cpu()], average='weighted')
        except ValueError:
                test_auc_acc_w=0
                pass
        pred = logits.argmax(1).cpu()
        labels_l = labels.argmax(1).cpu()
        train_acc = (pred[train_indices.cpu()] == labels_l[train_indices.cpu()]).float().mean()
        val_acc = (pred[valid_indices.cpu()] == labels_l[valid_indices.cpu()]).float().mean()
        test_acc = (pred[test_indices.cpu()] == labels_l[test_indices.cpu()]).float().mean()
        test_acc_w = 0

        macro_test, micro_test = macro_micro_f1(
            torch.max(labels[test_indices], 1)[1].cpu(), torch.max(logits[test_indices], 1)[1].cpu())
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 5 == 0:
            print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f), Weighted Test Acc %.4f' % (
                loss.item(),
                train_acc.item() if th.is_tensor(train_acc) else train_acc,
                val_acc.item()if th.is_tensor(val_acc) else val_acc,
                best_val_acc.item()if th.is_tensor(best_val_acc) else best_val_acc,
                test_acc.item()if th.is_tensor(test_acc) else test_acc,
                best_test_acc.item()if th.is_tensor(best_test_acc) else best_test_acc,test_acc_w.item()
                if th.is_tensor(test_acc_w) else test_acc_w
            ))
    print()
    return best_test_acc