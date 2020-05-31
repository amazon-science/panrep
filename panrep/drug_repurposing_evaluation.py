import csv

import numpy as np
import torch as th
import os,pickle
from model import PanRepHetero
from encoders import EncoderRelGraphConvHetero
from node_supervision_tasks import  LinkPredictor
from classifiers import End2EndLinkPredictorRGCN
from node_supervision_tasks import LinkPredictorLearnableEmbed
import argparse
from load_data import load_univ_hetero_data

def drugtreatsdisease_all_disease_types_as_one(model_name='model2020-05-18-07:58:54.540612',n_layers=1, n_hidden=800, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=10, dropout=0.1
                                               ):
    COV_disease_list = [
        'Disease::SARS-CoV2 E',
        'Disease::SARS-CoV2 M',
        'Disease::SARS-CoV2 N',
        'Disease::SARS-CoV2 Spike',
        'Disease::SARS-CoV2 nsp1',
        'Disease::SARS-CoV2 nsp10',
        'Disease::SARS-CoV2 nsp11',
        'Disease::SARS-CoV2 nsp12',
        'Disease::SARS-CoV2 nsp13',
        'Disease::SARS-CoV2 nsp14',
        'Disease::SARS-CoV2 nsp15',
        'Disease::SARS-CoV2 nsp2',
        'Disease::SARS-CoV2 nsp4',
        'Disease::SARS-CoV2 nsp5',
        'Disease::SARS-CoV2 nsp5_C145A',
        'Disease::SARS-CoV2 nsp6',
        'Disease::SARS-CoV2 nsp7',
        'Disease::SARS-CoV2 nsp8',
        'Disease::SARS-CoV2 nsp9',
        'Disease::SARS-CoV2 orf10',
        'Disease::SARS-CoV2 orf3a',
        'Disease::SARS-CoV2 orf3b',
        'Disease::SARS-CoV2 orf6',
        'Disease::SARS-CoV2 orf7a',
        'Disease::SARS-CoV2 orf8',
        'Disease::SARS-CoV2 orf9b',
        'Disease::SARS-CoV2 orf9c',
        'Disease::MESH:D045169',
        'Disease::MESH:D045473',
        'Disease::MESH:D001351',
        'Disease::MESH:D065207',
        'Disease::MESH:D028941',
        'Disease::MESH:D058957',
        'Disease::MESH:D006517'
    ]
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    g=splits_dir['train_g']
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_dictionary['Disease'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)


    metapathRWSupervision=None
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=g.etypes,
                                      ntype2id=ntype2id, use_cuda=False,edg_pct=1,ng_rate=1)

    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict={},
                             masked_node_types=g.ntypes,
                             loss_over_all_nodes=True,
                             use_infomax_task=False,
                             use_reconstruction_task=False,
                             use_node_motif=False,
                             link_predictor=link_predictor,
                             out_motif_dict={},
                             use_cluster=False,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision)
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model.encoder.forward(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Disease'][disease_ids]
    scores_per_disease = []
    dids = []
    drug_ids = th.tensor(drug_ids).long()

    rids = th.tensor(0).repeat((len(compound_embs),1))

    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        score=model.linkPredictor.calc_pos_score_with_rids(h_emb=disease_emb,t_emb=compound_embs,rids=rids)
        scores_per_disease.append(score)
        dids.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 1000
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    for i in range(topk):
        drug = entity_id_map[int(proposed_dids[i])][10:17]
        if clinical_drug_map.get(drug, None) is not None:
            score = proposed_scores[i]
            print("[{}]\t{}\t{}".format(i, clinical_drug_map[drug], score, proposed_scores[i]))

def drugtreatsdisease_per_disease_type(model_name='model2020-05-18-07:58:54.540612',n_layers=1, n_hidden=800, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=10, dropout=0.1,per_rel=False
                                               ):
    COV_disease_list = [
        'Disease::SARS-CoV2 E',
        'Disease::SARS-CoV2 M',
        'Disease::SARS-CoV2 N',
        'Disease::SARS-CoV2 Spike',
        'Disease::SARS-CoV2 nsp1',
        'Disease::SARS-CoV2 nsp10',
        'Disease::SARS-CoV2 nsp11',
        'Disease::SARS-CoV2 nsp12',
        'Disease::SARS-CoV2 nsp13',
        'Disease::SARS-CoV2 nsp14',
        'Disease::SARS-CoV2 nsp15',
        'Disease::SARS-CoV2 nsp2',
        'Disease::SARS-CoV2 nsp4',
        'Disease::SARS-CoV2 nsp5',
        'Disease::SARS-CoV2 nsp5_C145A',
        'Disease::SARS-CoV2 nsp6',
        'Disease::SARS-CoV2 nsp7',
        'Disease::SARS-CoV2 nsp8',
        'Disease::SARS-CoV2 nsp9',
        'Disease::SARS-CoV2 orf10',
        'Disease::SARS-CoV2 orf3a',
        'Disease::SARS-CoV2 orf3b',
        'Disease::SARS-CoV2 orf6',
        'Disease::SARS-CoV2 orf7a',
        'Disease::SARS-CoV2 orf8',
        'Disease::SARS-CoV2 orf9b',
        'Disease::SARS-CoV2 orf9c',
        'Disease::MESH:D045169',
        'Disease::MESH:D045473',
        'Disease::MESH:D001351',
        'Disease::MESH:D065207',
        'Disease::MESH:D028941',
        'Disease::MESH:D058957',
        'Disease::MESH:D006517'
    ]
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    g=splits_dir['train_g']
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_dictionary['Disease'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)


    metapathRWSupervision=None
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=g.etypes,
                                      ntype2id=ntype2id, use_cuda=False,edg_pct=1,ng_rate=1)

    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict={},
                             masked_node_types=g.ntypes,
                             loss_over_all_nodes=True,
                             use_infomax_task=False,
                             use_reconstruction_task=False,
                             use_node_motif=False,
                             link_predictor=link_predictor,
                             out_motif_dict={},
                             use_cluster=False,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision)
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model.encoder.forward(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Disease'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()

    rids = th.tensor(0).repeat((len(compound_embs),1))

    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        if per_rel:
            score = model.linkPredictor.calc_pos_score_with_rids_per_rel(h_emb=disease_emb, t_emb=compound_embs,
                                                                         rids=rids)
        else:
            score = model.linkPredictor.calc_pos_score_with_rids(h_emb=disease_emb, t_emb=compound_embs, rids=rids)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""
    max_dugs = ""
    drugsfr_in_top_k = {}
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)
    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)
def drugtreatsdisease_per_disease_type_pr(model_name='model2020-05-26-21:58:52.754973',n_layers=1, n_hidden=640, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=10, dropout=0.1,per_rel=False
                                               ):
    COV_disease_list = [
        'Disease::SARS-CoV2 E',
        'Disease::SARS-CoV2 M',
        'Disease::SARS-CoV2 N',
        'Disease::SARS-CoV2 Spike',
        'Disease::SARS-CoV2 nsp1',
        'Disease::SARS-CoV2 nsp10',
        'Disease::SARS-CoV2 nsp11',
        'Disease::SARS-CoV2 nsp12',
        'Disease::SARS-CoV2 nsp13',
        'Disease::SARS-CoV2 nsp14',
        'Disease::SARS-CoV2 nsp15',
        'Disease::SARS-CoV2 nsp2',
        'Disease::SARS-CoV2 nsp4',
        'Disease::SARS-CoV2 nsp5',
        'Disease::SARS-CoV2 nsp5_C145A',
        'Disease::SARS-CoV2 nsp6',
        'Disease::SARS-CoV2 nsp7',
        'Disease::SARS-CoV2 nsp8',
        'Disease::SARS-CoV2 nsp9',
        'Disease::SARS-CoV2 orf10',
        'Disease::SARS-CoV2 orf3a',
        'Disease::SARS-CoV2 orf3b',
        'Disease::SARS-CoV2 orf6',
        'Disease::SARS-CoV2 orf7a',
        'Disease::SARS-CoV2 orf8',
        'Disease::SARS-CoV2 orf9b',
        'Disease::SARS-CoV2 orf9c',
        'Disease::MESH:D045169',
        'Disease::MESH:D045473',
        'Disease::MESH:D001351',
        'Disease::MESH:D065207',
        'Disease::MESH:D028941',
        'Disease::MESH:D058957',
        'Disease::MESH:D006517'
    ]
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))
    parser = argparse.ArgumentParser(description='PanRep-FineTune')
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    args = parser.parse_args(['--dataset', 'drkg'])
    args.gpu=-1
    args.few_shot = False
    args.motif_clusters=5
    args.use_node_motifs=True
    args.rw_supervision = False
    train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, metapaths, \
    train_edges, test_edges, valid_edges, g, valid_g, test_g = \
        load_univ_hetero_data(args)
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_dictionary['Disease'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)


    metapathRWSupervision=None
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictor(out_dim=n_hidden, etypes=g.etypes,
                                   ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                   ng_rate=5)
    out_motif_dict = {}
    for name in g.ntypes:
        out_motif_dict[name] = g.nodes[name].data['motifs'].size(1)
    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict={},
                             masked_node_types=[],
                             loss_over_all_nodes=False,
                             use_infomax_task=True,
                             use_reconstruction_task=False,
                             use_node_motif=True,
                             link_predictor=link_predictor,
                             out_motif_dict=out_motif_dict,
                             use_cluster=5>0,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision, focus_category=False)
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path,map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model.encoder.forward(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Disease'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()

    rids = th.tensor(0).repeat((len(compound_embs)))
    etype2id = {}
    for i, etype in enumerate(g.etypes):
            etype2id[etype] = i
    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        if per_rel:
            score = model.linkPredictor.calc_pos_score_with_rids_per_rel(h_emb=disease_emb, t_emb=compound_embs,
                                                                         rids=rids)
        else:
            score = model.linkPredictor.calc_pos_score_with_rids(h_emb=disease_emb, t_emb=compound_embs,etypes2ids=etype2id, rids=rids)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""
    max_dugs = ""
    drugsfr_in_top_k = {}
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)
    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)
def druginhibitsgene_per_gene_type_pr(model_name='model2020-05-26-21:58:52.754973',n_layers=1, n_hidden=640, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=10, dropout=0.1,per_rel=False
                                               ):
    import pandas as pd
    import numpy as np
    file = '../data/drkg/drkg/coronavirus-related-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov_genes = np.unique(df.values[:, 2]).tolist()
    file = '../data/drkg/drkg/covid19-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov2_genes = np.unique(df.values[:, 2]).tolist()
    # keep unique related genes

    cov_related_genes = list(set(cov_genes + cov2_genes))
    # cov_related_genes=list(set(cov2_genes))
    print(len(cov_related_genes))
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))
    parser = argparse.ArgumentParser(description='PanRep-FineTune')
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    args = parser.parse_args(['--dataset', 'drkg'])
    args.gpu = -1
    args.few_shot = False
    args.motif_clusters = 5
    args.use_node_motifs = True
    args.rw_supervision = False
    train_idx, test_idx, val_idx, labels, category, num_classes, featless_node_types, metapaths, \
    train_edges, test_edges, valid_edges, g, valid_g, test_g = \
        load_univ_hetero_data(args)
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in cov_related_genes:
        disease_ids.append(entity_dictionary['Gene'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)


    metapathRWSupervision=None
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictor(out_dim=n_hidden, etypes=g.etypes,
                                   ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                   ng_rate=5)
    out_motif_dict = {}
    for name in g.ntypes:
        out_motif_dict[name] = g.nodes[name].data['motifs'].size(1)
    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict={},
                             masked_node_types=[],
                             loss_over_all_nodes=False,
                             use_infomax_task=True,
                             use_reconstruction_task=False,
                             use_node_motif=True,
                             link_predictor=link_predictor,
                             out_motif_dict=out_motif_dict,
                             use_cluster=5>0,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision, focus_category=False)
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model.encoder.forward(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Gene'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()

    rids = th.tensor(0).repeat((len(compound_embs)))
    etype2id = {}
    for i, etype in enumerate(g.etypes):
            etype2id[etype] = i
    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        if per_rel:
            score = model.linkPredictor.calc_pos_score_with_rids_per_rel(h_emb=disease_emb, t_emb=compound_embs, etypes2ids=etype2id, rids=rids)
        else:
            score=model.linkPredictor.calc_pos_score_with_rids(h_emb=disease_emb,t_emb=compound_embs,etypes2ids=etype2id,rids=rids)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""

    drugsfr_in_top_k = {}

    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)

def drugtreatsdisease_per_disease_type_rgcn(model_name='model2020-05-20-05:01:51.617645',n_layers=1, n_hidden=600, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=20, dropout=0.1,per_rel=False
                                               ):
    COV_disease_list = [
        'Disease::SARS-CoV2 E',
        'Disease::SARS-CoV2 M',
        'Disease::SARS-CoV2 N',
        'Disease::SARS-CoV2 Spike',
        'Disease::SARS-CoV2 nsp1',
        'Disease::SARS-CoV2 nsp10',
        'Disease::SARS-CoV2 nsp11',
        'Disease::SARS-CoV2 nsp12',
        'Disease::SARS-CoV2 nsp13',
        'Disease::SARS-CoV2 nsp14',
        'Disease::SARS-CoV2 nsp15',
        'Disease::SARS-CoV2 nsp2',
        'Disease::SARS-CoV2 nsp4',
        'Disease::SARS-CoV2 nsp5',
        'Disease::SARS-CoV2 nsp5_C145A',
        'Disease::SARS-CoV2 nsp6',
        'Disease::SARS-CoV2 nsp7',
        'Disease::SARS-CoV2 nsp8',
        'Disease::SARS-CoV2 nsp9',
        'Disease::SARS-CoV2 orf10',
        'Disease::SARS-CoV2 orf3a',
        'Disease::SARS-CoV2 orf3b',
        'Disease::SARS-CoV2 orf6',
        'Disease::SARS-CoV2 orf7a',
        'Disease::SARS-CoV2 orf8',
        'Disease::SARS-CoV2 orf9b',
        'Disease::SARS-CoV2 orf9c',
        'Disease::MESH:D045169',
        'Disease::MESH:D045473',
        'Disease::MESH:D001351',
        'Disease::MESH:D065207',
        'Disease::MESH:D028941',
        'Disease::MESH:D058957',
        'Disease::MESH:D006517'
    ]
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    g=splits_dir['train_g']
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in COV_disease_list:
        disease_ids.append(entity_dictionary['Disease'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
        ntype2id[ntype] = i

    link_predictor = LinkPredictor(out_dim=n_hidden, etypes=g.etypes,
                                   ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                   ng_rate=5, shared_rel_emb=False)

    model = End2EndLinkPredictorRGCN(
        h_dim=n_hidden,
        out_dim=n_hidden,
        num_rels=len(set(g.etypes)),
        rel_names=list(set(g.etypes)),
        num_bases=n_bases, g=g, device=device,
        num_hidden_layers=n_layers,
        dropout=dropout,
        use_self_loop=use_self_loop)
    model.link_predictor = link_predictor
    model_path = 'saved_model/' + model_name  #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model(g)

    compound_embs = embeddings['Compound'][drug_ids]
    disease_embs = embeddings['Gene'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()
    # GNBR::N::Compound:Gene is 27 relation
    rids = th.tensor(35).repeat((len(compound_embs)))
    etype2id = {}
    for i, etype in enumerate(g.etypes):
        etype2id[etype] = i
    if device != 'cpu':
        model = model.to(device)
        disease_embs = disease_embs.to(device)
        compound_embs = compound_embs.to(device)
    for did in range(len(disease_embs)):
        disease_emb = disease_embs[did]
        # for disease_id in disease_ids:
        # disease_emb = entity_emb[disease_id]
        # print(did)
        disease_emb = disease_emb.repeat((compound_embs.shape[0], 1))
        score = model.link_predictor.calc_pos_score_with_rids(h_emb=disease_emb, t_emb=compound_embs, rids=rids,
                                                              etypes2ids=etype2id, device=device)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""
    max_dugs = ""
    drugsfr_in_top_k = {}
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)
    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)


def druginhibitsgene_per_gene_type(model_name='model2020-05-18-07:58:54.540612',n_layers=1, n_hidden=800, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=10, dropout=0.1,per_rel=False
                                               ):
    import pandas as pd
    import numpy as np
    file = '../data/drkg/drkg/coronavirus-related-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov_genes = np.unique(df.values[:, 2]).tolist()
    file = '../data/drkg/drkg/covid19-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov2_genes = np.unique(df.values[:, 2]).tolist()
    # keep unique related genes

    cov_related_genes = list(set(cov_genes + cov2_genes))
    # cov_related_genes=list(set(cov2_genes))
    print(len(cov_related_genes))
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    g=splits_dir['train_g']
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in cov_related_genes:
        disease_ids.append(entity_dictionary['Gene'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    encoder=EncoderRelGraphConvHetero(
                                  n_hidden,
                                    g=g,
                                    device=device,
                                  num_bases=n_bases,
                                          etypes=g.etypes,
                                        ntypes=g.ntypes,
                                  num_hidden_layers=n_layers,
                                  dropout=dropout,
                                  use_self_loop=use_self_loop)


    metapathRWSupervision=None
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictorLearnableEmbed(out_dim=n_hidden, etypes=g.etypes,
                                      ntype2id=ntype2id, use_cuda=False,edg_pct=1,ng_rate=1)

    model = PanRepHetero(    n_hidden,
                             n_hidden,
                             etypes=g.etypes,
                             encoder=encoder,
                             ntype2id=ntype2id,
                             num_hidden_layers=n_layers,
                             dropout=dropout,
                             out_size_dict={},
                             masked_node_types=g.ntypes,
                             loss_over_all_nodes=True,
                             use_infomax_task=False,
                             use_reconstruction_task=False,
                             use_node_motif=False,
                             link_predictor=link_predictor,
                             out_motif_dict={},
                             use_cluster=False,
                             single_layer=False,
                             use_cuda=use_cuda,
                             metapathRWSupervision=metapathRWSupervision)
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)

    model.eval()

    with th.no_grad():
        embeddings = model.encoder.forward(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Gene'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()

    rids = th.tensor(0).repeat((len(compound_embs),1))

    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        if per_rel:
            score = model.linkPredictor.calc_pos_score_with_rids_per_rel(h_emb=disease_emb, t_emb=compound_embs, rids=rids)
        else:
            score=model.linkPredictor.calc_pos_score_with_rids(h_emb=disease_emb,t_emb=compound_embs,rids=rids)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)
    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""

    drugsfr_in_top_k = {}

    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)

def count_drugs_in_top100(scores_per_disease,dids_per_gene,entity_id_map,clinical_drug_map):
    drugsfr_in_top_k = {}

    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if drug in drugsfr_in_top_k:
                drugsfr_in_top_k[drug] += 1
            else:
                drugsfr_in_top_k[drug] = 1

    hits_across_drugs=list(drugsfr_in_top_k.values())
    name_drugs = list(drugsfr_in_top_k.keys())
    ids=np.argsort(hits_across_drugs)
    topk = 1000
    ids=ids[::-1]
    top_ids=ids[:topk]
    for  i  in range(len(top_ids)):
        top_id=top_ids[i]
        drug = name_drugs[top_id]
        if clinical_drug_map.get(drug, None) is not None:
            print("{}\t{}\t{}\t{}\n".format(drug, clinical_drug_map[drug],hits_across_drugs[top_id],i))
        #else:
        #    print("{}\t{}\t{}\n".format(drug,  hits_across_drugs[top_id], top_id))

def druginhibitsgene_per_gene_type_rgcn(model_name='model2020-05-20-05:01:51.617645',n_layers=1, n_hidden=600, use_self_loop=True,
                                               use_cuda=False, device="cpu", n_bases=20, dropout=0.1,per_rel=False
                                               ):
    import pandas as pd
    import numpy as np
    file = '../data/drkg/drkg/coronavirus-related-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov_genes = np.unique(df.values[:, 2]).tolist()
    file = '../data/drkg/drkg/covid19-host-genes.tsv'
    df = pd.read_csv(file, sep="\t")
    cov2_genes = np.unique(df.values[:, 2]).tolist()
    # keep unique related genes

    cov_related_genes = list(set(cov_genes + cov2_genes))
    # cov_related_genes=list(set(cov2_genes))
    print(len(cov_related_genes))
    data_folder = "../data/drkg/drkg/"
    splits_dir=pickle.load(open(os.path.join(data_folder, 'complete_splits_dir.pickle'), "rb"))

    g=splits_dir['train_g']
    # Load entity file
    drug_list = []
    with open(data_folder+"infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug', 'ids'])
        for row_val in reader:
            drug_list.append(row_val['drug'])

    treatment = ['Hetionet::CtD::Compound:Disease'] #'GNBR::T::Compound:Disease'
    # retrieve ID
    entity_ids={}
    relation_ids=g.etypes
    entity_dictionary=pickle.load(open(os.path.join("../data/drkg/drkg/", "drkg_entity_id_map.pickle"), "rb"))

    # Get drugname/disease name to entity ID mappings

    drug_ids = []
    disease_ids = []
    for drug in drug_list:
        drug_ids.append(entity_dictionary['Compound'][drug])

    for disease in cov_related_genes:
        disease_ids.append(entity_dictionary['Gene'][disease])
    entity_id_map={v:k for k,v in entity_dictionary['Compound'].items()
                   }
    #treatment_rid = [relation_map[treat] for treat in treatment]
    # calculate score for these triples
    ntype2id = {}
    for i, ntype in enumerate(g.ntypes):
            ntype2id[ntype] = i

    link_predictor = LinkPredictor(out_dim=n_hidden, etypes=g.etypes,
                                                         ntype2id=ntype2id, use_cuda=use_cuda, edg_pct=1,
                                                         ng_rate=5,shared_rel_emb=False)

    model = End2EndLinkPredictorRGCN(
                           h_dim=n_hidden,
                           out_dim=n_hidden,
                           num_rels = len(set(g.etypes)),
                            rel_names=list(set(g.etypes)),
                           num_bases=n_bases,g=g,device=device,
                           num_hidden_layers=n_layers,
                           dropout=dropout,
                           use_self_loop=use_self_loop)
    model.link_predictor=link_predictor
    model_path = 'saved_model/' + model_name #
    checkpoint = th.load(model_path)
    model.load_state_dict(checkpoint)


    model.eval()

    with th.no_grad():
        embeddings = model(g)

    compound_embs=embeddings['Compound'][drug_ids]
    disease_embs=embeddings['Gene'][disease_ids]
    scores_per_disease = []
    dids_per_gene = []
    drug_ids = th.tensor(drug_ids).long()
    #GNBR::N::Compound:Gene is 27 relation
    rids = th.tensor(27).repeat((len(compound_embs)))
    etype2id = {}
    for i, etype in enumerate(g.etypes):
            etype2id[etype] = i
    if device!='cpu':
        model=model.to(device)
        disease_embs=disease_embs.to(device)
        compound_embs=compound_embs.to(device)
    for did in range(len(disease_embs)):
        disease_emb=disease_embs[did]
        #for disease_id in disease_ids:
        #disease_emb = entity_emb[disease_id]
        #print(did)
        disease_emb=disease_emb.repeat((compound_embs.shape[0],1))
        score=model.link_predictor.calc_pos_score_with_rids(h_emb=disease_emb,t_emb=compound_embs,rids=rids,
                                                            etypes2ids=etype2id,device=device)
        scores_per_disease.append(score)
        dids_per_gene.append(drug_ids)
    scores = th.cat(scores_per_disease)

    dids = th.cat(dids_per_gene)

    # sort scores in decending order
    idx = th.flip(th.argsort(scores), dims=[0])
    scores = scores[idx].detach().numpy()
    dids = dids[idx].numpy()
    _, unique_indices = np.unique(dids, return_index=True)
    topk = 100
    topk_indices = np.sort(unique_indices)[:topk]
    proposed_dids = dids[topk_indices]
    proposed_scores = scores[topk_indices]
    # compare with clinical trial drugs

    clinical_drugs_file = data_folder+'COVID19_clinical_trial_drugs.tsv'
    clinical_drug_map = {}
    with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name', 'drug_id'])
        for row_val in reader:
            clinical_drug_map[row_val['drug_id']] = row_val['drug_name']

    maxhit = 0
    drugs_in_top_k = {}
    maxgene=""

    drugsfr_in_top_k = {}
    for i in range(len(scores_per_disease)):
        score = scores_per_disease[i]
        did = dids_per_gene[i]
        idx = th.flip(th.argsort(score), dims=[0])
        score = score[idx].detach().numpy()
        did = did[idx].detach().numpy()
        # print(did)
        _, unique_indices = np.unique(did, return_index=True)
        topk = 100
        topk_indices = np.sort(unique_indices)[:topk]
        proposed_did = did[topk_indices]
        proposed_score = score[topk_indices]
        found_in_top_k = 0
        found_drugs = "\n"
        for j in range(topk):
            drug = entity_id_map[int(proposed_did[j])][10:17]
            if clinical_drug_map.get(drug, None) is not None:
                found_in_top_k += 1
                score = proposed_score[j]
                if drug in drugs_in_top_k:
                    drugs_in_top_k[drug] += 1
                    drugsfr_in_top_k[drug] += 1 / (j + 1)
                else:
                    drugs_in_top_k[drug] = 1
                    drugsfr_in_top_k[drug] = 1 / (j + 1)
                found_drugs += "[{}]{}\n".format(j, clinical_drug_map[drug])
                # print("[{}]{}".format(j, clinical_drug_map[drug]))
        # print("{}\t{}".format(cov_related_genes[i], found_in_top_k))
        if maxhit < found_in_top_k:
            maxhit = found_in_top_k
            maxgene = scores_per_disease[i]
            max_dugs = found_drugs
    print("{}\t{}\t{}".format(maxgene, maxhit, max_dugs))

    res = [[drug, clinical_drug_map[drug], drugs_in_top_k[drug], drugsfr_in_top_k[drug]] for drug in
           drugs_in_top_k.keys()]
    res = reversed(sorted(res, key=lambda x: x[2]))
    for drug in res:
        print("{}\t{}\t{}\t{}".format(drug[0], drug[1], drug[2], drug[3]))
    count_drugs_in_top100(scores_per_disease, dids_per_gene, entity_id_map, clinical_drug_map)


if __name__ == '__main__':
    #
    #
    #drugtreatsdisease_per_disease_type_rgcn(model_name='model2020-05-21-00:44:53.901835', n_bases=10)
    #drugtreatsdisease_per_disease_type(model_name='model2020-05-18-07:27:13.309872', n_hidden=600)
    #druginhibitsgene_per_gene_type(model_name='model2020-05-18-07:27:13.309872',n_hidden=600)
    #drugtreatsdisease_per_disease_type(model_name='model2020-05-19-03:20:20.517401',n_hidden=600,n_bases=20)
    #druginhibitsgene_per_gene_type(model_name='model2020-05-19-03:20:20.517401',n_hidden=600,n_bases=20)
    #
    drugtreatsdisease_per_disease_type_pr(model_name='fin_model2020-05-27-11:44:05.706244',n_bases=20)
    druginhibitsgene_per_gene_type_pr(model_name='fin_model2020-05-27-11:44:05.706244',n_bases=20)
    #druginhibitsgene_per_gene_type_rgcn(model_name='model2020-05-21-00:44:53.901835',n_bases=10)
    #druginhibitsgene_per_gene_type_rgcn(n_layers=0)
    #drugtreatsdisease_per_disease_type(model_name='model2020-05-18-07:27:13.309872',n_hidden=600)
    #drugtreatsdisease_per_disease_type(model_name='model2020-05-18-07:08:10.379638', n_hidden=400)


