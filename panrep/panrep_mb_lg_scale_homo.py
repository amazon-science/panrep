"""
PanRep: Universal node embeddings for heterogeneous graphs
Paper:
Code:

"""
from evaluation import evaluation_link_prediction, \
    direct_eval_lppr_link_prediction, macro_micro_f1, evaluate_results_nc, \
    mlp_classifier,compute_acc
from node_sampling_masking import  NegativeSampler,NeighborSampler,NeighborEdgeSampler
import os
from utils import calculate_entropy
from datetime import datetime
import pickle
import argparse
from layers import          RelGraphEmbedLayerHomo
import dgl
from dgl.nn import RelGraphConv
import numpy as np
import tqdm
import copy
import time
import torch
from classifiers import ClassifierMLP
from load_data import load_univ_homo_data,generate_rwalks
from model import PanRepHomo
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from decoders import MetapathRWalkerSupervision, \
    LinkPredictor,LinkPredictorLearnableEmbed, MutualInformationDiscriminatorHomo,LinkPredictorHomoLS,ClusterRecoverDecoderHomo
from encoders import EncoderRelGraphConvHomo,EncoderHGT
import torch.nn.functional as F


def finetune_panrep_fn_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, batch_size,
                                           n_hidden, negative_rate_lp, fanout, l2norm,
                                           n_layers, n_lp_fintune_epochs, lr_lp_ft, use_cuda, device, learn_rel_embed=False):
    ntype2id = {}
    for i, ntype in enumerate(train_g.ntypes):
            ntype2id[ntype] = i
    if learn_rel_embed:
        finetune_link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=train_g.etypes,
                                                     ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1, negative_rate_lp=negative_rate_lp)
    else:
        finetune_link_predictor = LinkPredictor(out_dim=n_hidden, etypes=train_g.etypes,
                                       ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                       negative_rate_lp=negative_rate_lp)

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


        for i, (seeds, blocks) in enumerate(loader):
            loss = finetune_lp_model.link_predictor_forward_mb(p_blocks=blocks)
            finetune_lp_optimizer.zero_grad()
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

def eval_training_of_panrep(model, dataloader):
    model.eval()

    print("==========Start evaluating===============")
    for i, (seeds, blocks) in enumerate(dataloader):

        loss, embeddings = model.forward_mb(p_blocks=blocks)
    print("=============Evaluation finished=============")
    return

def initiate_model(args, g,node_feats,num_rels):

    out_size_dict={}
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    for name in g.ntypes:
        if g.nodes[name].data.get("h_f", None) is not None:
            if args.num_cluster>0:
                out_size_dict[name]=g.nodes[name].data['h_clusters'].size(1)
            else:
                out_size_dict[name] = g.nodes[name].data['h_f'].size(1)
        else:
            out_size_dict[name] =0

    # for the motif layer

    out_motif_dict = {}
    if args.use_node_motifs:
        for name in g.ntypes:
            out_motif_dict[name] = g.nodes[name].data['motifs'].size(1)

    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i
    if args.encoder=="RGCN":
        encoder=EncoderRelGraphConvHomo(
                         device=device,
                         num_nodes=g.number_of_nodes,
                         h_dim=args.n_hidden,
                         num_rels=num_rels,
                         num_bases=args.n_bases,
                         num_hidden_layers=args.n_layers,
                         dropout=args.dropout,
                         use_self_loop=args.use_self_loop,low_mem=args.low_mem)

    elif args.encoder=="HGT":
        raise ValueError('Not implemented')
        node_dict = {}
        edge_dict = {}
        for ntype in g.ntypes:
            node_dict[ntype] = len(node_dict)
        for etype in g.etypes:
            edge_dict[etype] = len(edge_dict)
            g.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
        encoder = EncoderHGT(g,
                             node_dict, edge_dict,
                             n_inp=args.n_inp,
                             n_hid=args.n_hid,
                             n_out=labels.max().item() + 1,
                             n_layers=2,
                             n_heads=4,
                             use_norm=True).to(device)
    decoders=torch.nn.ModuleDict()
    if args.rw_supervision:
        mrw_interact = {}
        for ntype in g.ntypes:
            mrw_interact[ntype]=[]
            for neighbor_ntype in g.ntypes:
                mrw_interact[ntype] +=[neighbor_ntype]
        decoders['mrwd']= MetapathRWalkerSupervision(in_dim=args.n_hidden, negative_rate=args.negative_rate_rw,
                                                           device=device,mrw_interact=mrw_interact).to(device)

    learn_rel_embed=False
    if args.use_link_prediction:
        if learn_rel_embed:
            link_predictor = LinkPredictorLearnableEmbedHomo(out_dim=args.n_hidden, etypes=g.etypes,
                                                         ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1, negative_rate_lp=args.negative_rate_lp).to(device)
        else:
            link_predictor = LinkPredictorHomoLS(
                h_dim=args.n_hidden, num_rels=num_rels, use_cuda=use_cuda).to(device)
        decoders['lpd']=link_predictor
    if args.use_infomax_loss:
        decoders['mid']=MutualInformationDiscriminatorHomo(n_hidden=args.n_hidden).to(device)
    if args.use_node_motifs:
        decoders['nmd']=NodeMotifDecoderHomo(in_dim=args.n_hidden, h_dim=args.n_hidden, out_dict=out_motif_dict).to(device)
    if args.use_clusterandrecover_loss:
        decoders['crd']=ClusterRecoverDecoderHomo(
            n_cluster=args.num_cluster, in_size=args.n_hidden,
            h_dim=args.n_hidden).to(device)


    node_tids = g.ndata[dgl.NTYPE]

    embed_layer = RelGraphEmbedLayerHomo(args.gpu  ,
                                     g.number_of_nodes(),
                                     node_tids,
                                     len(node_feats),
                                     node_feats,
                                     args.n_hidden,
                                     sparse_emb=args.sparse_embedding)
    model = PanRepHomo(encoder=encoder,
                         decoders=decoders,embed_layer=embed_layer)
    return model

def evaluate_pr(model, eval_loader, node_feats):
    model.eval()
    eval_logits = []
    eval_seeds = []

    with torch.no_grad():
        for sample_data in tqdm.tqdm(eval_loader):
            torch.cuda.empty_cache()
            seeds, blocks = sample_data
            logits = model.obtain_embeddings(blocks, node_feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)

    return eval_logits, eval_seeds

def evaluate_prft(model, eval_loader, node_feats,forward_function):
    model.eval()
    eval_logits = []
    eval_seeds = []

    with torch.no_grad():
        for sample_data in tqdm.tqdm(eval_loader):
            torch.cuda.empty_cache()
            seeds, blocks = sample_data
            logits = forward_function(blocks,node_feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = torch.cat(eval_logits)
    eval_seeds = torch.cat(eval_seeds)

    return eval_logits, eval_seeds






def finetune_panrep_for_node_classification(args, device, labels, metapaths,
                                            model, multilabel, num_classes,
                                              test_idx, g,
                                            train_idx, use_cuda, val_idx,target_idx,node_feats,num_rels=None):
    # add one layer for the input layer
#    remove_decoders=True
#    if remove_decoders:
#        model.decoders={}
    classifier_b = 'rgcn'
    if classifier_b=='rgcn':
        fanouts = [args.fanout] * (args.n_layers + 2)
        classifier=RelGraphConv(
            args.n_hidden, num_classes, num_rels, "basis",
            args.n_bases, activation=None,
            self_loop=args.use_self_loop,
            low_mem=args.low_mem)
        forward_function=model.classifier_forward_rgcn
    else:
        fanouts = [args.fanout]*(args.n_layers+1)
        classifier=ClassifierMLP(input_size=args.n_hidden, hidden_size=args.n_hidden, out_size=num_classes,
                      single_layer=args.single_layer)
        forward_function=model.classifier_forward
    sampler = NeighborSampler(g, target_idx, fanouts)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=args.num_workers)

    # validation sampler
    val_sampler = NeighborSampler(g, target_idx, fanouts)
    val_loader = DataLoader(dataset=val_idx.numpy(),
                            batch_size=args.eval_batch_size,
                            collate_fn=val_sampler.sample_blocks,
                            shuffle=False,
                            num_workers=args.num_workers)

    # validation sampler
    test_sampler = NeighborSampler(g, target_idx, fanouts)
    test_loader = DataLoader(dataset=test_idx.numpy(),
                             batch_size=args.eval_batch_size,
                             collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=args.num_workers)
    # set up fine_tune epochs
    # donwstream classifier for model supervision

    model.classifier = classifier
    if multilabel is False:
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        if args.klloss:
            loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        else:
            loss_func = torch.nn.BCEWithLogitsLoss()
    if use_cuda:
        model.cuda()
        model.classifier.cuda()
    labels = labels  # .float()
    best_test_acc = 0
    best_val_acc = 0
    lbl = labels
    adjust_pr_lr = False  # to add a slower learning rate for parameters of the model compared to decoder parameters.
    if adjust_pr_lr:
        cl_params = set(list(model.classifier.parameters()))
        tot_params = set(list(model.parameters()))
        res_params = list(tot_params.difference(cl_params))
        optimizer = torch.optim.Adam([{'params': res_params},
                                      {'params': model.classifier.parameters(), 'lr': args.lr}],
                                     lr=args.lr / 10, weight_decay=args.l2norm)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    # training loop
    print("start finetuning training...")

    model.train()
    dur = []
    # cancel the link preidction and rw superivsion tasks since the batch graph sampled will not have
    # connections among different nodes. Consider full graph sampling but evaluate only in the category node.
    model.link_prediction_task = False
    model.rw_supervision_task = False

    for epoch in range(args.n_fine_tune_epochs):
        model.train()

        if epoch > 3:
            t0 = time.time()

        for i, (seeds, blocks) in enumerate(loader):
            batch_tic = time.time()
            # need to copy the features
            for j in range(len(blocks)):

                blocks[j] = blocks[j].to(device)
            lbl = labels[seeds]
            logits=forward_function(blocks, node_feats)
            if args.klloss:
                logits=torch.log_softmax(logits.squeeze(), dim=-1)

            log_loss = loss_func(logits, lbl)
            if args.only_ssl:
                loss = log_loss
            else:
                pr_loss = model(p_blocks=blocks,node_feats=node_feats, rw_neighbors=None)
                loss = pr_loss + log_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = (logits)
            if multilabel is False:
                pred = pred.argmax(dim=1)
            else:
                if args.klloss and multilabel:
                    pred = torch.log_softmax(pred.squeeze(), dim=-1)
            train_acc = compute_acc(pred,lbl,multilabel)

            if i%50==0:
                print(
                "Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
                    format(epoch, i, train_acc, loss.item(), time.time() - batch_tic))

        if epoch > 3:
            dur.append(time.time() - t0)

        val_logits, val_seeds=evaluate_prft(model, val_loader, node_feats,forward_function)
        if args.klloss and multilabel:
            val_logits = torch.log_softmax(val_logits.squeeze(), dim=-1)
        val_loss = loss_func(val_logits, labels[val_seeds].cpu()).item()
        val_acc = compute_acc((val_logits),labels[val_seeds],multilabel)
        print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_loss, np.average(dur)))
    print()

    test_logits, test_seeds=evaluate_prft(model, test_loader, node_feats,forward_function)
    if args.klloss and multilabel:
        test_logits = torch.log_softmax(test_logits.squeeze(), dim=-1)
    test_acc = compute_acc((test_logits), labels[test_seeds], multilabel)


    return model, test_acc

def initialize_sampler(g, batch_size, args, target_idx, evaluate_panrep =False):
    if args.use_link_prediction:
        return initialize_sampler_lp_(g, batch_size, args,target_idx, evaluate_panrep )
    full_node_list = torch.arange(g.number_of_nodes())
    target_uns = torch.arange(g.number_of_nodes())
    val_pct = 0.1

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    # add one for the input layer
    fanouts = [args.fanout]*(args.n_layers+1)
    sampler = NeighborSampler(g, full_node_list, fanouts)



    if evaluate_panrep:
        pr_node_ids=list(full_node_list.numpy())
        pr_val_ind=list(np.random.choice(len(pr_node_ids), int(len(pr_node_ids)*val_pct), replace=False))
        pr_train_ind=list(set(list(np.arange(len(pr_node_ids)))).difference(set(pr_val_ind)))
        loader = DataLoader(dataset=pr_train_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=0)


        valid_loader = DataLoader(dataset=pr_val_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=0)
    else:
        loader = DataLoader(dataset=target_uns.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=0)

        valid_loader=None
    # validation sampler

    test_sampler = NeighborSampler(g, full_node_list, fanouts)
    # the loader must obtain embeddings from all nodes
    test_loader = DataLoader(dataset=target_idx.numpy(),
                                 batch_size=args.eval_batch_size,
                                 collate_fn=test_sampler.sample_blocks,
                                 shuffle=False,
                                 num_workers=0)

    return loader,valid_loader,test_loader


def initialize_sampler_lp_(g, batch_size, args, target_idx, evaluate_panrep =False):
    """ When lp is used different loader is required """
    full_node_list = torch.arange(g.number_of_nodes())
    target_uns = torch.arange(g.number_of_nodes())
    val_pct = 0.1

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    # add one for the input layer
    fanouts = [args.fanout]*(args.n_layers+1)
    sampler = NeighborSampler(g, full_node_list, fanouts)
    n_edges = g.num_edges()
    train_seeds = np.arange(n_edges)
    sampler_edge = NeighborEdgeSampler(sampler=dgl.dataloading.MultiLayerNeighborSampler(
        fanouts))


    if evaluate_panrep:

            pr_node_ids=list(train_seeds)
            pr_val_ind=list(np.random.choice(len(pr_node_ids), int(len(pr_node_ids)*val_pct), replace=False))
            pr_train_ind=list(set(list(np.arange(len(pr_node_ids)))).difference(set(pr_val_ind)))
            loader = dgl.dataloading.EdgeDataLoader(
             g, pr_train_ind, sampler_edge, exclude='reverse_id',reverse_eids=torch.cat([
             torch.arange(n_edges // 2, n_edges),
             torch.arange(0, n_edges // 2)]),negative_sampler=NegativeSampler(g, args.negative_rate_lp),
             batch_size=args.batch_size,            shuffle=True,            drop_last=False,
             pin_memory=True,num_workers=args.num_workers)


            valid_loader = DataLoader(dataset=pr_val_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=0)
    else:
        loader = dgl.dataloading.EdgeDataLoader(
            g, list(train_seeds), sampler_edge, exclude='reverse_id', reverse_eids=torch.cat([
                torch.arange(n_edges // 2, n_edges),
                torch.arange(0, n_edges // 2)]), negative_sampler=NegativeSampler(g, args.negative_rate_lp),
            batch_size=args.batch_size, shuffle=True, drop_last=False,
            pin_memory=True, num_workers=args.num_workers)

        valid_loader=None
    # validation sampler

    test_sampler = NeighborSampler(g, full_node_list, fanouts)
    # the loader must obtain embeddings from all nodes
    test_loader = DataLoader(dataset=target_idx.numpy(),
                                 batch_size=args.eval_batch_size,
                                 collate_fn=test_sampler.sample_blocks,
                                 shuffle=False,
                                 num_workers=0)

    return loader,valid_loader,test_loader
def _fit(args):
    #dblp add the original node split and test...
    if args.rw_supervision and args.n_layers==1:
        return "Not supported"
    if args.num_cluster>0 and args.use_clusterandrecover_loss:
        args.use_clusterandrecover_loss=True
    else:
        args.use_clusterandrecover_loss = False

    train_idx,val_idx,test_idx,target_idx,labels,num_classes,node_feats,cluster_assignments,\
               metapaths, train_edges, test_edges, valid_edges, train_g, valid_g, test_g,multilabel,num_rels=load_univ_homo_data(args)
    # sampler parameters
    eval_nc=labels is not None
    svm_eval=False
    eval_lp = args.test_edge_split > 0

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    # create model


    model=initiate_model(args,train_g,node_feats,num_rels)


    if use_cuda:
        model.cuda()
        labels = labels.cuda()
        #train_g = train_g.to(device)


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # sampler for minibatch
    evaluate_panrep_decoders_during_training = False
    evaluate_every = 20
    loader,valid_loader,test_loader=initialize_sampler(train_g, args.batch_size, args, target_idx, evaluate_panrep_decoders_during_training)
    # training loop
    print("start pretraining...")
    for epoch in range(args.n_epochs):
        model.train()
        rw_neighbors = generate_rwalks(g=train_g, metapaths=metapaths, samples_per_node=4, device=device,rw_supervision=args.rw_supervision)


        for i, batch in enumerate(loader):
            if args.use_link_prediction:
                (input_nodes, pos_graph, neg_graph, blocks)=batch
            else:
                (input_nodes, blocks)=batch
                pos_graph=None
                neg_graph=None
            loss = model(p_blocks=blocks,node_feats=node_feats,cluster_assignments=cluster_assignments,
                         rw_neighbors=rw_neighbors,graphs=(pos_graph,neg_graph))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100==0:
                print("Train Loss: {:.4f} Epoch {:05d} | Batch {:03d}".format(loss.item(), epoch, i))
        if evaluate_panrep_decoders_during_training and epoch % evaluate_every == 0:
            eval_training_of_panrep(model=model, dataloader=valid_loader)
    print("end pretraining...")


    ## Obtain universal embeddings and evaluate
    if args.n_epochs>0:
        universal_embeddings, test_seeds =  evaluate_pr(model, test_loader, node_feats)
        idx_sorted=torch.argsort(test_seeds)
        universal_embeddings=universal_embeddings[idx_sorted]
    else:
        universal_embeddings=None
    # calculate entropy
    entropy = ""

    ## Evaluate link prediction and finetune for linkprediction
    pr_mrr="PanRep "

    if eval_lp:
        if args.use_link_prediction:
            # Evaluated LP model in PanRep for link prediction
            pr_mrr+= direct_eval_lppr_link_prediction(test_g, model, train_edges, valid_edges, test_edges, args.n_hidden, args.n_layers,
                                                      eval_neg_cnt=100, use_cuda=True)
        # finetune PanRep for link prediction
        n_lp_fintune_epochs=args.n_epochs
        lr_lp_ft=args.lr
        pr_mrr+=finetune_panrep_fn_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, args.batch_size,
                                                       args.n_hidden, args.negative_rate_lp, args.fanout, args.l2norm,
                                                       args.n_layers, n_lp_fintune_epochs, lr_lp_ft, use_cuda, device)

    if eval_nc:



        if multilabel:
            labels_i=np.argmax(labels.cpu().numpy(),axis=1)
        else:
            labels_i=labels.cpu().numpy()
        ## Test universal embeddings by training mlp classifier for node classification
        if args.n_epochs>0:
            test_acc_prembed = mlp_classifier(args,
            universal_embeddings, use_cuda, args.num_hidden_downstream, args.lr_downstream, args.num_epochs_downstream, multilabel, num_classes, labels,
            train_idx, val_idx, test_idx, device, batch_size=args.batch_size_downstream)
        else:
            test_acc_prembed=0
        if svm_eval:
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,macro_str,micro_str = evaluate_results_nc(
                universal_embeddings[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
        else:
            macro_str=""
            micro_str=""

        ## Finetune PanRep for node classification
        model,test_acc = finetune_panrep_for_node_classification(args, device,
                                                                                      labels,
                                                                                      metapaths, model, multilabel,
                                                                                      num_classes,
                                                                                      test_idx, train_g,
                                                                                      train_idx, use_cuda,
                                                                                             val_idx,target_idx,
                                                                                                      node_feats,num_rels=num_rels)


        universal_embeddings, test_seeds =  evaluate_pr(model, test_loader, node_feats)
        idx_sorted=torch.argsort(test_seeds)
        universal_embeddings=universal_embeddings[idx_sorted]
        print("Test accuracy: {:4f}".format(test_acc))


        evaluate_prft_embeddings=False
        if evaluate_prft_embeddings:
            ## Test finetuned embeddings by training an mlp classifier
            test_acc_ftembed= mlp_classifier(args,
                universal_embeddings, use_cuda, args.num_hidden_downstream, args.lr_downstream, args.num_epochs_downstream, multilabel, num_classes, labels, train_idx, val_idx, test_idx, device)

            if svm_eval:
                if multilabel:
                    labels_i = np.argmax(labels.cpu().numpy(), axis=1)
                else:
                    labels_i = labels.cpu().numpy()
                svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,finmacro_str,finmicro_str = evaluate_results_nc(
                    universal_embeddings[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
        else:
            finmacro_str=""
            finmicro_str=""
            test_acc_ftembed=""
        return " | PanRep " +macro_str+" "+micro_str+" | Test acc  PanRepFT: {:4f} | ".format(test_acc)+\
               " " \
        "| Finetune "+finmacro_str+" "+finmicro_str+"PR MRR : "+pr_mrr+\
              " Entropy "+ str(entropy) + " Test acc PR +MLP "+str(test_acc_prembed)+\
               " Test acc PRft+MLP "+str(test_acc_ftembed)
    return "PR MRR : " + pr_mrr + " Entropy " + str(entropy)



def fit(args):
        '''
            best results 700 hidden units so far
        '''
        args.splitpct = 0.1
        n_epochs_list = [0]#[250,300]
        n_hidden_list =[128,256,300]#[40,200,400]
        n_layers_list = [0,1]
        n_fine_tune_epochs_list= [100]#[20,50]#[30,50,150]
        n_bases_list = [-1,2]
        lr_list = [5e-3]
        dropout_list = [0.5]
        fanout_list = []

        test_edge_split_list = [-0.025]
        use_link_prediction_list = [False]
        use_clusterandrecover_loss_list =[False]#[False,True]
        use_infomax_loss_list = [True]#[False,True]
        use_node_motif_list = [False]
        rw_supervision_list=[False]
        num_cluster_list=[5]
        K_list=[0]#[2,5,10,15]
        motif_cluster_list=[5]#[2,6,10]
        use_self_loop_list=[True]
        negative_rate_lp_list=[5]
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
                                                for use_clusterandrecover_loss in use_clusterandrecover_loss_list:
                                                        for use_self_loop in use_self_loop_list:
                                                            for use_node_motif in use_node_motif_list:
                                                                for num_cluster in num_cluster_list:
                                                                    for single_layer in single_layer_list:
                                                                        for motif_cluster in motif_cluster_list:
                                                                            for rw_supervision in rw_supervision_list:
                                                                                for k_fold in K_list:
                                                                                    for negative_rate_lp in negative_rate_lp_list:
                                                                                        for only_ssl in only_ssl_list:
                                                                                            if not use_clusterandrecover_loss and not \
                                                                                                    use_infomax_loss and not use_link_prediction\
                                                                                                    and not use_node_motif and not rw_supervision:
                                                                                                continue
                                                                                            else:
                                                                                                fanout=25
                                                                                                args.rw_supervision = rw_supervision
                                                                                                args.n_layers = n_layers
                                                                                                args.use_clusterandrecover_loss = use_clusterandrecover_loss
                                                                                                args.num_cluster = num_cluster
                                                                                                args.use_node_motifs = use_node_motif
                                                                                                args.k_fold = k_fold
                                                                                                args.test_edge_split = test_edge_split
                                                                                                args.use_link_prediction = use_link_prediction
                                                                                                args.motif_clusters = motif_cluster
                                                                                                args.use_infomax_loss = use_infomax_loss
                                                                                                args.n_hidden = n_hidden
                                                                                                args.n_bases = n_bases
                                                                                                args.dropout = dropout
                                                                                                args.fanout = fanout
                                                                                                args.use_self_loop = use_self_loop
                                                                                                args.negative_rate_lp = negative_rate_lp
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
                                                                                                         use_clusterandrecover_loss,
                                                                                                         use_infomax_loss,
                                                                                                         use_self_loop,
                                                                                                         use_node_motif,
                                                                                                         num_cluster,
                                                                                                         single_layer,
                                                                                                         motif_cluster,
                                                                                                         k_fold,
                                                                                                         rw_supervision,
                                                                                                         negative_rate_lp,
                                                                                                         only_ssl,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PanRep-FineTune')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=60,
            help="number of hidden units") # use 16, 2 for debug
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--num-cluster", type=int, default=6,
            help="number of clusters considered in the cluster and recover supervision if 0 then the "
                 "attribute reconstruction method is used by default")
    parser.add_argument("--motif-clusters", type=int, default=6,
            help="number of clusters considered in the node motif supervision")
    parser.add_argument("--test-edge-split", type=float, default=0,
            help="Pct of edges considered for the link prediction task. If 0 then the  task is skipped.")
    parser.add_argument("--splitpct", type=float, default=0.1,
                        help="Pct of nodes considered for the node classification task. If 0 then the task is skipped.")
    parser.add_argument("--eval-batch-size", type=int, default=128,
                            help="Mini-batch size. ")


    parser.add_argument("--n-bases", type=int, default=20,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=3,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=700,
            help="number of training epochs for PanRep")
    parser.add_argument("-n_fine_tune_epochs", "--n-fine-tune-epochs", type=int, default=3,
            help="number of training epochs for PanRep-FT ")
    parser.add_argument('--num-epochs-downstream', type=int, default=10)
    parser.add_argument('--num-hidden-downstream', type=int, default=256)
    parser.add_argument('--batch_size_downstream', type=int, default=500)
    parser.add_argument('--lr-downstream', type=float, default=0.01)
    parser.add_argument("-negative_rate_rw", "--negative-rate-rw", type=int, default=4,
                        help="number of negative examples per positive link for metapathrw supervision")
    parser.add_argument("-negative_rate_lp", "--negative-rate-lp", type=int, default=5,
                        help="number of negative examples per positive link for link prediction supervision")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("-en", "--encoder", type=str, required=True,
                        help="Encoder to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")

    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--use-infomax-loss", default=True, action='store_true',
                        help="use infomax task supervision")
    parser.add_argument("--use-clusterandrecover-loss", default=False, action='store_true',
                        help="use feature reconstruction task supervision")
    parser.add_argument("--loss-over-all-nodes", default=True, action='store_true',
                        help="compute the feature reconstruction loss over all nods or just the masked")
    parser.add_argument("--use-link-prediction", default=False, action='store_true',
                       help="use link prediction as supervision task")
    parser.add_argument("--rw-supervision", default=False, action='store_true',
                       help="use mrw supervision as supervision task")
    parser.add_argument("--only-ssl", default=True, action='store_true',
                       help="use only the semi-supervised loss while finetuning or add it to the rest of the losses.")
    parser.add_argument("--single-layer", default=True, action='store_true',
                       help="use only a single layer for the decoder for the semi-supervised loss.")
    parser.add_argument("--use_node_features", default=True, action='store_true',
                       help="use node features or use embeddings instead.")
    parser.add_argument("--use-node-motifs", default=True, action='store_true',
                       help="use the node motifs")
    parser.add_argument('--global-norm', default=False, action='store_true',
                help='User global norm instead of per node type norm')

    parser.add_argument("--num-workers", type=int, default=0,
                help="Number of workers for dataloader.")

    parser.add_argument("--low-mem", default=True, action='store_true',
               help="Whether use low mem RelGraphCov")
    parser.add_argument("--klloss", default=True, action='store_true',
               help="Whether use klloss in multilabel")
    parser.add_argument("--batch-size", type=int, default=1024,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--fanout", type=int, default=10,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--split", type=int, default=0,
                        help="split type.")
    parser.add_argument("--sparse-embedding", action='store_true', default=False,help='Use sparse embedding for node embeddings.')
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args(['--dataset', 'oag_cs','--encoder', 'RGCN'])


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