import torch
import numpy as np
import time
def create_edge_mask(old_g,use_cuda):
    g=old_g.local_var()
    for etype in g.etypes:
        if use_cuda:
            g.edges[etype].data['mask']=torch.tensor(np.ones((g.number_of_edges(etype))),dtype=torch.float32).cuda()
        else:
            g.edges[etype].data['mask'] = torch.tensor(np.ones((g.number_of_edges(etype))),dtype=torch.float32)
    return g
def unmask_edges(old_g,use_cuda):
    return create_edge_mask(old_g,use_cuda)

def negative_sampling(g,pos_samples_d,  negative_rate):
    labels_d={}
    samples_d={}
    for e in pos_samples_d.keys():
        (s,e,o)=g.to_canonical_etype(e)
        num_entity_s=g.number_of_nodes(s)
        num_entity_o=g.number_of_nodes(o)
        pos_samples=np.stack(pos_samples_d[e])#.transpose()
        size_of_batch = len(pos_samples)
        num_to_generate = size_of_batch * negative_rate
        neg_samples = np.tile(pos_samples, (negative_rate, 1))
        labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
        labels[: size_of_batch] = 1
        values_s = np.random.randint(num_entity_s, size=num_to_generate)
        values_o = np.random.randint(num_entity_o, size=num_to_generate)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values_s[subj]
        neg_samples[obj, 1] = values_o[obj]
        samples_d[e] = np.concatenate((pos_samples, neg_samples))
        labels_d[e] = labels

    return samples_d, labels_d


def hetero_edge_masker_sampler_mb(old_g, pct_masked_edges, negative_rate, edge_masking,batch_pct, use_cuda):
    '''
    This function masks some edges at random by setting the mask attribute to 0
    :param old_g: The original graph
    :param pct_masked_edges: The pct of edges to sample
    :param negative_rate: The nbr of negative samples per edge
    :param edge_masking: Whether to mask the sampled edges or not
    :param batch_pct: The pct of edges in the sampled graph with respect to the original graph
    :return: The altered graph, the positive and the negative samples.
    '''
    pos_samples = {}
    neg_samples = {}
    g = old_g.local_var()
    for etype in g.etypes:
        pos_samples[etype]=[]
        u,v,eid=g.all_edges(form='all', etype=etype)
        lnum_sampled_edges = int(batch_pct * len(eid))
        sampl_ids = np.random.choice(g.number_of_edges(etype), size=lnum_sampled_edges, replace=False)
        u_retain=u[sampl_ids]
        v_retain=v[sampl_ids]
        eid_retain=eid[sampl_ids]
        pos_samples[etype] = np.stack((u_retain, v_retain).transpose())
        # mask edges
        # TODO 1. Make sure the graph is connected (how) ? rgcn does not
        #      2. Make sure negatives are true negatives
        #      3.
        if edge_masking:
            lnum_masked_edges = int(pct_masked_edges * lnum_sampled_edges)
            mask_ids = np.random.choice(lnum_sampled_edges, size=lnum_masked_edges, replace=False)
            sampled_edges = eid_retain[mask_ids]
            u_for_g = u_retain[mask_ids]
            v_for_g = v_retain[mask_ids]
        else:
            u_for_g = u_retain
            v_for_g = v_retain

    # TODO negative samples
    t0 = time.time()
    samples_d,labels_d=negative_sampling(g,pos_samples, negative_rate)
    tnega = time.time() - t0
    link_labels_d = {}
    for e in labels_d.keys():
        link_labels_d[e] = torch.from_numpy(labels_d[e])
        if use_cuda:
            link_labels_d[e] = link_labels_d[e].cuda()
    llabels_d = link_labels_d
    # create function consuming pos
    return g,samples_d,llabels_d

def hetero_edge_masker_sampler(old_g, pct_masked_edges, negative_rate, edge_masking, use_cuda):
    '''
    This function masks some edges at random by setting the mask attribute to 0
    :param old_g: The original graph
    :param pct_masked_edges: The pct of edges to sample
    :param negative_rate: The nbr of negative samples per edge
    :param edge_masking: Whether to mask the sampled edges or not
    :return: The altered graph, the positive and the negative samples.
    '''
    pos_samples = {}
    neg_samples = {}
    g = old_g.local_var()
    for etype in g.etypes:
        pos_samples[etype]=[]
        t0=time.time()
        u,v,eid=g.all_edges(form='all', etype=etype)
        tedgefetch=time.time()-t0

        uniq_v, edges = np.unique((u.data.numpy(), v.data.numpy()), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, dst)).transpose()

        # negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            negative_rate)
        pos_samples[etype]=np.stack((u,v)).transpose()



        # mask edges
        if edge_masking:
            lnum_sampled_edges = int(pct_masked_edges * len(eid))
            sampl_ids = np.random.choice(g.number_of_edges(etype), size=lnum_sampled_edges, replace=False)
            sampled_edges = eid[sampl_ids]
            g.edges[etype].data['mask'][sampled_edges] = 0
    # TODO negative samples
    t0 = time.time()
    samples_d,labels_d=negative_sampling(g,pos_samples, negative_rate)
    tnega = time.time() - t0
    link_labels_d = {}
    for e in labels_d.keys():
        link_labels_d[e] = torch.from_numpy(labels_d[e])
        if use_cuda:
            link_labels_d[e] = link_labels_d[e].cuda()
    llabels_d = link_labels_d
    # create function consuming pos
    return g,samples_d,llabels_d