"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""

import argparse,gc

import time
import torch
import torch.nn.functional as F

from load_data import load_hetero_link_pred_data
from torch.utils.data import DataLoader
from classifiers import DLinkPredictorMB as DownstreamLinkPredictorMB
from edge_masking_samling import RGCNLinkRankSampler

def main(args):
    best_mrr, best_result=fit(args)
    print('Best results')
    print(best_result)

def _fit(n_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,args):
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_edges, test_edges, valid_edges, train_g, valid_g, test_g, featless_node_types=load_hetero_link_pred_data(args)
    num_rels = len(list(train_edges.keys()))
    valid_set = valid_edges
    train_set = train_edges
    test_set = test_edges

    # hyper params
    fanout = fanout
    batch_size = int(32*1024)
    chunk_size = 32
    use_self_loop = True
    regularization_coef = 0.0001
    train_grad_clip = 1.0
    fanouts = [fanout] * n_layers
    edge_masking=True
    # this are tripplets containing src id rel id and dest id

    num_edges = 0
    etype_map = {}
    etype_key_map = {}
    etypes = []
    head_ids = []
    tail_ids = []
    eids = []
    i = 0
    # key is (src_type, label, dst_type)
    for key, val in train_set.items():
        etype_map[i] = key
        etype_key_map[key] = i
        n_edges = val[0].shape[0]
        etypes.append(torch.full((n_edges,), i))
        head_ids.append(val[0])
        tail_ids.append(val[1])
        num_edges += n_edges
        i += 1

    print("Total number of edges {}/{}".format(num_edges, i))
    etypes = torch.cat(etypes, dim=0)
    head_ids = torch.cat(head_ids, dim=0)
    tail_ids = torch.cat(tail_ids, dim=0)
    etypes.share_memory_()
    head_ids.share_memory_()
    tail_ids.share_memory_()
    pos_seed = torch.arange((num_edges // batch_size) * batch_size)
    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    sampler = RGCNLinkRankSampler(train_g,
                                  num_edges,
                                  etypes,
                                  etype_map,
                                  head_ids,
                                  tail_ids,
                                  fanouts,
                                  device=device,
                                  nhead_ids=head_ids,
                                  ntail_ids=tail_ids,edge_masking=edge_masking)
    dataloader = DataLoader(dataset=pos_seed,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            num_workers=0)


    # map input features as embeddings
    # build input layer
    model = DownstreamLinkPredictorMB(test_g,
                                      args.gpu,
                                      n_hidden,
                                      num_rels,
                                      num_bases=n_bases,
                                      num_hidden_layers=n_layers,
                                      dropout=dropout,
                                      use_self_loop=use_self_loop,
                                      regularization_coef=regularization_coef,etype_key_map=etype_key_map)

    if use_cuda:
        model.cuda()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training loop
    print("start training...")
    dur = []
    for epoch in range(n_epochs):
        model.train()
        if epoch > 3:
            t0 = time.time()
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data
            for i in range(len(n_blocks)):
                n_blocks[i] = n_blocks[i].to(device)
            for i in range(len(p_blocks)):
                p_blocks[i] = p_blocks[i].to(device)

            p_h= model(p_blocks)
            n_h=model(n_blocks)

            # loss calculation
            loss = model.get_loss(p_h, p_g, n_h,n_g,
                                  int(batch_size / chunk_size),
                                  chunk_size,
                                  chunk_size)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_grad_clip)
            t1 = time.time()

            print("Epoch {}, Iter {}, Loss:{}".format(epoch, i, loss.detach()))

        if epoch > 3:
            dur.append(t1 - t0)
        p_g = None
        n_g = None
        p_blocks = None
        n_blocks = None
        gc.collect()
        _eval_mrr(model, valid_g, [train_set, valid_set], valid_set, neg_cnt=1000)
    return model

def fit(args):
        n_epochs_list = [1,400,600]
        n_hidden_list = [40,400,600]
        n_layers_list = [2]
        n_bases_list = [30]
        lr_list = [1e-4,1e-5]
        dropout_list = [0.2]
        fanout_list = [None]

        rgcn_results = []
        best_mrr = .0
        best_model = None
        best_result = None
        for n_epochs in n_epochs_list:
            for n_hidden in n_hidden_list:
                for n_layers in n_layers_list:
                    for n_bases in n_bases_list:
                        for fanout in fanout_list:
                            for lr in lr_list:
                                for dropout in dropout_list:
                                    model, test_g, train_set, valid_set,  test_set=_fit(n_epochs, n_layers, n_hidden, n_bases, fanout, lr, dropout,args)
                                    mrr = _eval(model, test_g, train_set, valid_set,  test_set)
                                    result = "RGCN Model, n_epochs {}; n_hidden {}; n_layers {}; n_bases {}; " \
                                            "fanout {}; lr {}; dropout {} acc {}".format(n_epochs,
                                                                                         n_hidden,
                                                                                         n_layers,
                                                                                         n_bases,
                                                                                         0,
                                                                                         lr,
                                                                                         dropout,
                                                                                         mrr)
                                    print(result)
                                rgcn_results.append(result)
                                if mrr > best_mrr:
                                    best_model = model
                                    best_result = result
                                    best_mrr = best_mrr

        _all_results = rgcn_results
        _best_model = best_model
        _best_result = best_result
        _best_mrr = best_mrr
        return best_mrr, best_result

def _eval_mrr(model, test_g, data_sets, test_set, neg_cnt=1000):
        etype_map = {}
        etype_key_map = {}
        num_edges = 0
        etypes = []
        head_ids = []
        tail_ids = []
        i = 0
        # key is (src_type, label, dst_type)
        for key, val in data_sets[0].items():
            etype_map[i] = key
            etype_key_map[key] = i
            i += 1

        for data_set in data_sets:
            for key, val in data_set.items():
                i = etype_key_map[key]
                n_edges = val[0].shape[0]
                etypes.append(torch.full((n_edges,), i))
                head_ids.append(val[0])
                tail_ids.append(val[1])
                num_edges += n_edges
        print("Total number of edges {}".format(num_edges))
        etypes = torch.cat(etypes, dim=0)
        head_ids = torch.cat(head_ids, dim=0)
        tail_ids = torch.cat(tail_ids, dim=0)
        etypes.share_memory_()
        head_ids.share_memory_()
        tail_ids.share_memory_()

        num_test_edges = 0
        test_head_ids = []
        test_tail_ids = []
        for key, val in test_set.items():
            n_edges = val[0].shape[0]
            test_head_ids.append(val[0])
            test_tail_ids.append(val[1])
            num_test_edges += n_edges
        print("Total number of edges for eval {}".format(num_test_edges))
        seed = torch.arange(num_test_edges)
        test_head_ids = torch.cat(test_head_ids, dim=0)
        test_tail_ids = torch.cat(test_tail_ids, dim=0)
        test_head_ids.share_memory_()
        test_tail_ids.share_memory_()

        batch_size = 1000
        fanouts = [20] * model.num_hidden_layers
        # check cuda
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.set_device(args.gpu)
        device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
        eval_sampler = RGCNLinkRankSampler(test_g,
                                           num_edges,
                                           etypes,
                                           etype_map,
                                           test_head_ids,
                                           test_tail_ids,
                                           fanouts,
                                   device=device, nhead_ids=head_ids,
                                           ntail_ids=tail_ids,
                                           num_neg=neg_cnt)
        dataloader = DataLoader(dataset=seed,
                                batch_size=batch_size,
                                collate_fn=eval_sampler.sample_blocks,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False,
                                num_workers=0)

        model.eval()
        # training loop
        print("start evaluating...")
        logs = []
        for i, sample_data in enumerate(dataloader):
            bsize, p_g, n_g, p_blocks, n_blocks = sample_data

            p_h = model(p_blocks)
            n_h = model(n_blocks)
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
                idx = etype_key_map[canonical_etype]
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

            pos_score = model.calc_pos_score(p_head_emb, p_tail_emb, rids)
            t_neg_score = model.calc_neg_tail_score(p_head_emb,
                                                    n_tail_emb,
                                                    rids,
                                                    1,
                                                    bsize,
                                                    neg_cnt)
            h_neg_score = model.calc_neg_head_score(n_head_emb,
                                                    p_tail_emb,
                                                    rids,
                                                    1,
                                                    bsize,
                                                    neg_cnt)
            pos_scores = F.logsigmoid(pos_score).reshape(bsize, -1)
            t_neg_score = F.logsigmoid(t_neg_score).reshape(bsize, neg_cnt)
            h_neg_score = F.logsigmoid(h_neg_score).reshape(bsize, neg_cnt)
            neg_scores = torch.cat([h_neg_score, t_neg_score], dim=1)
            rankings = torch.sum(neg_scores >= pos_scores, dim=1) + 1
            rankings = rankings.cpu().detach().numpy()
            for idx in range(bsize):
                ranking = rankings[idx]
                logs.append({
                    'MRR': 1.0 / ranking,
                    'MR': float(ranking),
                    'HITS@1': 1.0 if ranking <= 1 else 0.0,
                    'HITS@3': 1.0 if ranking <= 3 else 0.0,
                    'HITS@10': 1.0 if ranking <= 10 else 0.0
                })
            if i % 100 == 0:
                print("Eval {}/{}".format(i, num_test_edges // batch_size))
            gc.collect()

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        for k, v in metrics.items():
            print('Test average {}: {}'.format(k, v))

        return metrics['MRR']
    
    
def _eval(model, test_g, train_set, valid_set,  test_set):

        return _eval_mrr(model, test_g, [train_set, valid_set, test_set], test_set)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep')
    parser.add_argument("--dropout", type=float, default=0.1,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=100,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=7,
            help="gpu")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--lr", type=float, default=5e-3,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--regularization", type=float, default=0,
            help="regularization weight")
    parser.add_argument("-e", "--n-epochs", type=int, default=2000,
            help="number of training epochs for decoder")
    parser.add_argument("-ec", "--n-cepochs", type=int, default=6000,
                        help="number of training epochs for classification")
    parser.add_argument("-num_masked", "--n-masked-nodes", type=int, default=100,
                        help="number of masked nodes")
    parser.add_argument("-num_masked_links", "--n-masked-links", type=int, default=1000,
                        help="number of masked links")
    parser.add_argument("-negative_rate", "--negative-rate", type=int, default=10,
                        help="number of negative examples per masked link")
    parser.add_argument("-negative_rate_d", "--negative-rate-downstream", type=int, default=10,
                        help="number of negative examples per masked link for the downstream task")

    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("-en", "--encoder", type=str, required=True,
                        help="Encoder to use")
    parser.add_argument("--l2norm", type=float, default=0,
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
                        help="mask as subset of node features")
    parser.add_argument("--loss-over-all-nodes", default=False, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--link-prediction", default=True, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--mask-links", default=True, action='store_true',
                       help="mask the links to be predicted")
    parser.add_argument("--evaluate-every", type=int, default=500,
            help="perform evaluation every n epochs")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'query_biodata','--encoder', 'RGCN'])
    print(args)
    args.bfs_level = args.n_layers + 1 # pruning used nodes for memory
    main(args)
