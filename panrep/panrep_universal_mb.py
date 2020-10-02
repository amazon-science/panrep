"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn

Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
from link_prediction_evaluation import evaluation_link_prediction, \
    direct_eval_lppr_link_prediction
from node_classification_evaluation import macro_micro_f1, evaluate_results_nc, mlp_classifier
from node_sampling_masking import  HeteroNeighborSampler,InfomaxNodeRecNeighborSampler
import os
from utils import calculate_entropy
from datetime import datetime
import pickle
import argparse

import numpy as np

import copy
import time
import torch
from classifiers import ClassifierMLP
from load_data import load_univ_hetero_data,generate_rwalks
from model import PanRepHetero
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from node_supervision_tasks import MetapathRWalkerSupervision, LinkPredictor,LinkPredictorLearnableEmbed
from encoders import EncoderRelGraphConvHetero


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


def finetune_pr_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, batch_size,
                                       n_hidden,  ng_rate,fanout, l2norm,
                                       n_layers, n_lp_fintune_epochs,lr_lp_ft, use_cuda, device,learn_rel_embed=False):
    ntype2id = {}
    for i, ntype in enumerate(train_g.ntypes):
            ntype2id[ntype] = i
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

def eval_training_of_panrep(model, dataloader):
    model.eval()

    print("==========Start evaluating===============")
    for i, (seeds, blocks) in enumerate(dataloader):

        loss, embeddings = model.forward_mb(p_blocks=blocks)
    print("=============Evaluation finished=============")
    return

def initiate_model(args,masked_node_types,train_g):
    # for the embedding layer
    rw_supervision = args.rw_supervision
    n_layers = args.n_layers
    use_reconstruction_loss = args.use_reconstruction_loss
    num_cluster = args.num_cluster
    use_node_motif = args.use_node_motif
    use_link_prediction = args.use_link_prediction
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
    out_size_dict={}
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
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
    loss_over_all_nodes = True
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
    return model
def initiate_sampler(train_g,batch_size,args,evaluate_panrep =False):
    n_layers = args.n_layers
    fanout = args.fanout
    val_pct = 0.1

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    sampler = InfomaxNodeRecNeighborSampler(train_g, [fanout] * (n_layers), device=device)

    if evaluate_panrep:
        pr_node_ids=list(sampler.hetero_map.keys())
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
        pr_train_ind=list(sampler.hetero_map.keys())
        loader = DataLoader(dataset=pr_train_ind,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            num_workers=0)
        valid_loader=None
    return loader,valid_loader
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
    n_hidden = args.n_hidden
    fanout = args.fanout
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
    batch_size = 16*1024
    l2norm=0.0001

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)


    device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
    # create model


    model=initiate_model(args,masked_node_types,train_g)


    if use_cuda:
        model.cuda()
        labels = labels.cuda()
        #train_g = train_g.to(device)


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)

    # sampler for minibatch
    evaluate_panrep = False
    evaluate_every = 20
    loader,valid_loader=initiate_sampler(train_g,batch_size,args,evaluate_panrep)
    # training loop
    print("start training...")

    for epoch in range(n_epochs):
        model.train()
        rw_neighbors = generate_rwalks(g=train_g, metapaths=metapaths, samples_per_node=4, device=device,rw_supervision=rw_supervision)

        optimizer.zero_grad()
        for i, (seeds, blocks) in enumerate(loader):

            loss, embeddings = model.forward_mb(p_blocks=blocks,rw_neighbors=rw_neighbors)
            loss.backward()
            optimizer.step()

            print("Train Loss: {:.4f} Epoch {:05d} | Batch {:03d}".format(loss.item(), epoch, i))
        if evaluate_panrep and epoch % evaluate_every == 0:
            eval_training_of_panrep(model=model, dataloader=valid_loader)


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
        #train_g=train_g.to(device)
    #calculate entropy
    entropy = calculate_entropy(embeddings)
    print("Entropy: "+str(entropy))
    # evaluate link prediction
    pr_mrr="PanRep "
    eval_lp=args.test_edge_split>0
    if eval_lp:
        if use_link_prediction:
            # Evaluated LP model in PanRep for link prediction
            pr_mrr+= direct_eval_lppr_link_prediction(test_g, model, train_edges, valid_edges, test_edges, n_hidden, n_layers,
                                                      eval_neg_cnt=100, use_cuda=True)
        # finetune PanRep for link prediction
        n_lp_fintune_epochs=n_epochs
        lr_lp_ft=lr
        pr_mrr+=finetune_pr_for_link_prediction(train_g, test_g, train_edges, valid_edges, test_edges, model, batch_size,
                                        n_hidden,  ng_rate, fanout, l2norm,
                                        n_layers, n_lp_fintune_epochs, lr_lp_ft, use_cuda, device)


    eval_nc=True
    svm_eval=True
    if eval_nc and labels is not None:
        multilabel = True
        feats = embeddings[category]
        labels_i=np.argmax(labels.cpu().numpy(),axis=1)
        lr_d=lr
        n_cepochs=600#n_epochs
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
                            num_workers=0)


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

                macro_test,micro_test= macro_micro_f1(
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
        test_acc = roc_auc_score(labels[test_idx].cpu()
                                 , logits[test_idx], average='macro')
        print("Test accuracy: {:4f}".format(acc))
        print("Mean forward time: {:4f}".format(np.mean(forward_time[len(forward_time) // 4:])))
        print("Mean backward time: {:4f}".format(np.mean(backward_time[len(backward_time) // 4:])))

        feats = embeddings[category]
        test_acc_ftembed= mlp_classifier(
            feats, use_cuda, n_hidden, lr_d, n_cepochs, multilabel, num_classes, labels, train_idx, val_idx, test_idx, device)
        labels_i=np.argmax(labels.cpu().numpy(),axis=1)
        if svm_eval:
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std,finmacro_str,finmicro_str = evaluate_results_nc(
                feats[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
            print("With logits")
            svm_macro_f1_list, svm_micro_f1_list, nmi_mean, nmi_std, ari_mean, ari_std, macro_str_log, micro_str_log = evaluate_results_nc(
                logits[test_idx].cpu().numpy(), labels_i[test_idx], num_classes=num_classes)
        else:
            finmacro_str=""
            finmicro_str=""
            macro_str_log=""
            micro_str_log=""
        return " | PanRep " +macro_str+" "+micro_str+ " | Test accuracy: {:4f} | ".format(test_acc)+" " \
        "| Finetune "+finmacro_str+" "+finmicro_str+"PR MRR : "+pr_mrr+" Logits: "+macro_str_log+" "+\
               micro_str_log +" Entropy "+ str(entropy) + " Test acc PR embed "+str(test_acc_prembed)+\
               " Test acc PRft embed "+str(test_acc_ftembed)
    else:
        return"PR MRR : " + pr_mrr + " Entropy " + str(entropy)


def fit(args):
        '''
        TODO
            best results 700 hidden units so far
        '''
        args.splitpct = 0.1
        n_epochs_list = [800]#[250,300]
        n_hidden_list =[300]#[40,200,400]
        n_layers_list = [1]
        n_fine_tune_epochs_list= [140]#[20,50]#[30,50,150]
        n_bases_list = [10]
        lr_list = [1e-3]
        dropout_list = [0.1]
        focus=False
        fanout_list = [None]
        test_edge_split_list = [-0.025]
        use_link_prediction_list = [False]
        use_reconstruction_loss_list =[True]#[False,True]
        use_infomax_loss_list =[False]#[False,True]
        use_node_motif_list = [False]
        rw_supervision_list=[False]
        num_cluster_list=[0,2,4,6,10]
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
                                                                                                fanout=None
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

    args = parser.parse_args(['--dataset', 'imdb_preprocessed','--encoder', 'RGCN'])
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