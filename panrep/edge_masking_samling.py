import torch
import numpy as np
import time
import dgl
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
        #values_o = np.random.randint(num_entity_o, size=num_to_generate) TODO add later
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        neg_samples[subj, 0] = values_s[subj]
        neg_samples[obj, 1] = values_s[obj]
        samples_d[e] = np.concatenate((pos_samples, neg_samples))
        labels_d[e] = labels

    return samples_d, labels_d


class RGCNLinkRankSampler:
    def __init__(self, g, num_edges, etypes, etype_map, phead_ids, ptail_ids, fanouts,
                 nhead_ids, ntail_ids, num_neg=None,device=None):
        self.g = g
        self.num_edges = num_edges
        self.etypes = etypes
        self.etype_map = etype_map
        self.phead_ids = phead_ids
        self.ptail_ids = ptail_ids
        self.nhead_ids = nhead_ids
        self.ntail_ids = ntail_ids
        self.fanouts = fanouts
        self.num_neg = num_neg
        self.device=device

    def sample_blocks(self, seeds):
        block_sample_s=time.time()
        bsize = len(seeds)
        pseed = seeds
        if self.num_neg is not None:
            nseed = torch.randint(self.num_edges, (self.num_neg,))
        else:
            nseed = torch.randint(self.num_edges, (bsize,))
        g = self.g
        etypes = self.etypes
        etype_map = self.etype_map
        fanouts = self.fanouts
        phead_ids = self.phead_ids
        ptail_ids = self.ptail_ids
        nhead_ids = self.nhead_ids
        ntail_ids = self.ntail_ids

        device=self.device
        # positive seeds
        pseed = torch.stack(pseed)
        p_etypes = etypes[pseed]
        phead_ids = phead_ids[pseed]
        ptail_ids = ptail_ids[pseed]
        # negative seeds
        # Negative examples should be perturbed, here it does not seem that they are
        n_etypes = etypes[nseed]
        nhead_ids = nhead_ids[nseed]
        ntail_ids = ntail_ids[nseed]

        p_edges = {}
        p_subg = []
        for key, canonical_etypes in etype_map.items():
            pe_loc = (p_etypes == key)
            # extract the ids corresponding to the specific canonical type
            p_head = phead_ids[pe_loc]
            p_tail = ptail_ids[pe_loc]
            if p_head.shape[0] == 0:
                continue
            # input the edges in a dictionary of edges
            p_edges[canonical_etypes] = (p_head, p_tail)
            if canonical_etypes[0] == canonical_etypes[2]:
                # positive subgraph (of the same node type)
                p_subg.append(dgl.graph((p_head, p_tail),
                                        canonical_etypes[0],
                                        canonical_etypes[1],
                                        g.number_of_nodes(canonical_etypes[0])))
            else:
                # positive subgraph with different node types
                p_subg.append(dgl.bipartite((p_head, p_tail),
                                            utype=canonical_etypes[0],
                                            etype=canonical_etypes[1],
                                            vtype=canonical_etypes[2],
                                            card=(g.number_of_nodes(canonical_etypes[0]),
                                                  g.number_of_nodes(canonical_etypes[2]))))
        n_subg = []
        # build the negative subgraphs
        for key, canonical_etypes in etype_map.items():
            ne_loc = (n_etypes == key)
            n_head = nhead_ids[ne_loc]
            n_tail = ntail_ids[ne_loc]
            if n_head.shape[0] == 0:
                continue

            if canonical_etypes[0] == canonical_etypes[2]:
                n_subg.append(dgl.graph((n_head, n_tail),
                                        canonical_etypes[0],
                                        canonical_etypes[1],
                                        g.number_of_nodes(canonical_etypes[0])))
            else:
                n_subg.append(dgl.bipartite((n_head, n_tail),
                                            utype=canonical_etypes[0],
                                            etype=canonical_etypes[1],
                                            vtype=canonical_etypes[2]),
                              card=(g.number_of_nodes(canonical_etypes[0]),
                                    g.number_of_nodes(canonical_etypes[2])))
        # build the heterograph from the subgraphs
        p_g = dgl.hetero_from_relations(p_subg)
        n_g = dgl.hetero_from_relations(n_subg)
        p_g = dgl.compact_graphs(p_g)
        n_g = dgl.compact_graphs(n_g)

        pg_seed = {}
        ng_seed = {}
        # obtain the node internal ID to map back to the full HeteroGraph
        # The original IDs are preserved because they always keep the total number of nodes in the subgraph
        # the same as in the original graph
        for ntype in p_g.ntypes:
            pg_seed[ntype] = p_g.nodes[ntype].data[dgl.NID]
        for ntype in n_g.ntypes:
            ng_seed[ntype] = n_g.nodes[ntype].data[dgl.NID]

        p_blocks = []
        n_blocks = []
        p_cur = pg_seed
        n_cur = ng_seed
        frontier_s=time.time()
        for i, fanout in enumerate(fanouts):
            if fanout is None:
                p_frontier = dgl.in_subgraph(g, p_cur)
                n_frontier = dgl.in_subgraph(g, n_cur)
            else:
                p_frontier = dgl.sampling.sample_neighbors(g, p_cur, fanout)
                n_frontier = dgl.sampling.sample_neighbors(g, n_cur, fanout)
            # all the positive edges are removed among the positive seeds nodes in the first layer
            if i == 0 and len(p_edges) > 0:
                # remove edges here
                edge_to_del = {}

                for canonical_etype, pairs in p_edges.items():
                    eid_to_del = p_frontier.edge_ids(pairs[0],
                                                     pairs[1],
                                                     force_multi=True,
                                                     etype=canonical_etype)[2]

                    if eid_to_del.shape[0] > 0:
                        '''
                        if p_frontier.number_of_edges(canonical_etype) == eid_to_del.shape[0]:
                            print("All will be deleted {} {}/{}, skip".format(canonical_etype,
                                                                              eid_to_del.shape[0], 
                                                                              p_frontier.number_of_edges(canonical_etype)))
                            continue
                        '''
                        edge_to_del[canonical_etype] = eid_to_del
                old_frontier = p_frontier
                p_frontier = dgl.remove_edges(old_frontier, edge_to_del)

            p_block = dgl.to_block(p_frontier, p_cur)
            p_cur = {}
            for ntype in p_block.srctypes:
                p_cur[ntype] = p_block.srcnodes[ntype].data[dgl.NID]
            p_blocks.insert(0, p_block)

            n_block = dgl.to_block(n_frontier, n_cur)
            n_cur = {}
            for ntype in n_block.srctypes:
                n_cur[ntype] = n_block.srcnodes[ntype].data[dgl.NID]
            n_blocks.insert(0, n_block)
        # add features to block nodes in first layer only ?
        frontier_time=time.time()-frontier_s
        cops=time.time()
        for ntype in p_blocks[0].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                p_blocks[0].srcnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    p_blocks[0].srcnodes[ntype].data['_ID']]
        for ntype in n_blocks[0].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                n_blocks[0].srcnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    n_blocks[0].srcnodes[ntype].data['_ID']]
        time_copy=time.time()-cops

        for i in range(len(n_blocks)):
            n_blocks[i] = n_blocks[i].to(device)
        for i in range(len(p_blocks)):
            p_blocks[i] = p_blocks[i].to(device)
        block_sample_time=time.time()-block_sample_s
        #print('copy time')
        #print(time_copy)
        #print('frontier calculation time')
        #print(frontier_time)
        #print('overal sampling time')
        #print(block_sample_time)

        return (bsize, p_g, n_g, p_blocks, n_blocks)


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
        '''
        uniq_v, edges = np.unique((u.data.numpy(), v.data.numpy()), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, dst)).transpose()

        # negative sampling
        samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                            negative_rate)
        
        '''
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