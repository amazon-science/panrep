'''
This file contains functions to sample nodes and mask node attributes.

'''

import random
import copy
import torch
import numpy as np
import dgl
import time
import itertools


def node_masker(old_g, num_nodes, masked_node_types,node_masking,use_reconstruction_loss):
    masked_nodes={}
    if not use_reconstruction_loss:
        return masked_nodes,old_g
    g=old_g.local_var()

    for ntype in g.ntypes:
        mnnodes=num_nodes
        if ntype not in masked_node_types:

            g.nodes[ntype].data['masked_values'] = copy.deepcopy(g.nodes[ntype].data['features'])
            if node_masking:
                if mnnodes > g.nodes[ntype].data['features'].shape[0]:
                    mnnodes = g.nodes[ntype].data['features'].shape[0] // 3
                masked_ids = np.random.choice(g.number_of_nodes(ntype), size=mnnodes, replace=False)

                masked_nodes[ntype] = masked_ids
                new_val=torch.zeros((mnnodes,g.nodes[ntype].data['features'].shape[1]))
                if g.nodes[ntype].data['features'].is_cuda:
                    new_val=new_val.cuda()
                g.nodes[ntype].data['features'][masked_ids,:] = new_val

    return masked_nodes,g

def node_masker_mb(embeddings, num_nodes, masked_node_types,node_masking):
    masked_nodes={}
    if not node_masking:
        return masked_nodes,embeddings
    else:
        raise NotImplementedError
    altered_embeddings = copy.deepcopy(embeddings)

    for ntype in embeddings:
        mnnodes=num_nodes
        if ntype not in masked_node_types:
            if mnnodes>embeddings[ntype].shape[0]:
                mnnodes=embeddings[ntype].shape[0]//3
            masked_ids = np.random.choice(embeddings[ntype].shape[0], size=mnnodes, replace=False)

            masked_nodes[ntype]=masked_ids
            altered_embeddings[ntype][masked_ids,:]=torch.zeros((mnnodes,embeddings[ntype].shape[1]))

    return masked_nodes,altered_embeddings

class HeteroNeighborSampler:
    """Neighbor sampler on heterogeneous graphs
    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph
    category : str
        Category name of the seed nodes.
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, category, fanouts,device):
        self.g = g
        self.category = category
        self.fanouts = fanouts
        self.device=device

    def sample_blocks(self, seeds):
        blocks = []
        g=self.g
        seeds = {self.category : torch.tensor(seeds).long()}
        cur = seeds
        for fanout in self.fanouts:
            if fanout is None:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)
            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        for ntype in blocks[0].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                blocks[0].srcnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    blocks[0].srcnodes[ntype].data['_ID']]

        return seeds, blocks

class LinkPredictorEvalSampler:
    def __init__(self, g, fanouts, device, full_neighbor=False,category=None):
        """
        if fanouts is None, sample full neighbor
        """
        self.g = g
        self.fanouts = fanouts
        self.device=device
        self.full_neighbor = full_neighbor
        self.category = category
        #self.pct_batch=pct_batch
        if self.fanouts[0] is None:
            self.full_neighbor = True


        counter = itertools.count(0)
        self.hetero_map = {next(counter):(i,ntype)    for ntype in (g.ntypes) for i in range(g.number_of_nodes(ntype))}
        self.number_of_nodes=len(self.hetero_map)

    def sample_blocks(self, seeds_list):
        blocks = []
        seeds={}
        device = self.device
        g = self.g
        block_sample_s = time.time()
        if self.category is None:
            seeds_list.sort()
            for s in seeds_list:
                nid,ntype=self.hetero_map[s]
                if ntype not in seeds:
                    seeds[ntype]=[nid]
                else:
                    seeds[ntype]+=[nid]
            for ntype in seeds:
                seeds[ntype]=torch.tensor(seeds[ntype])#.to(device)
        else:
            seeds = {self.category: torch.tensor(seeds_list).long()}
        cur = seeds
        frontier_time_s=time.time()
        for fanout in self.fanouts:
            if self.full_neighbor:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)

            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        frontier_time=time.time()-frontier_time_s
        # add features to block nodes in first layer only ?

        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)

        block_sample_time=time.time()-block_sample_s
        #print('copy time')
        #print(time_copy)
        #print('frontier calculation time')
        #print(frontier_time)
        #print('overal sampling time')
        #print(block_sample_time)
        return seeds, blocks

class InfomaxNodeRecNeighborSampler:
    def __init__(self, g, fanouts, device, full_neighbor=False,category=None):
        """
        if fanouts is None, sample full neighbor
        """
        self.g = g
        self.fanouts = fanouts
        self.device=device
        self.full_neighbor = full_neighbor
        self.category = category
        #self.pct_batch=pct_batch
        if self.fanouts[0] is None:
            self.full_neighbor = True


        counter = itertools.count(0)
        self.hetero_map = {next(counter):(i,ntype)    for ntype in (g.ntypes) for i in range(g.number_of_nodes(ntype))}
        self.number_of_nodes=len(self.hetero_map)

    def sample_blocks(self, seeds_list):
        blocks = []
        seeds={}
        device = self.device
        g = self.g
        block_sample_s = time.time()
        if self.category is None:
            seeds_list.sort()
            for s in seeds_list:
                nid,ntype=self.hetero_map[s]
                if ntype not in seeds:
                    seeds[ntype]=[nid]
                else:
                    seeds[ntype]+=[nid]
            for ntype in seeds:
                seeds[ntype]=torch.tensor(seeds[ntype])#.to(device)
        else:
            seeds = {self.category: torch.tensor(seeds_list).long()#.to(device)
                     }
        cur = seeds
        frontier_time_s=time.time()
        for fanout in self.fanouts:
            if self.full_neighbor:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            block = dgl.to_block(frontier, cur)

            cur = {}
            for ntype in block.srctypes:
                cur[ntype] = block.srcnodes[ntype].data[dgl.NID]
            blocks.insert(0, block)
        frontier_time=time.time()-frontier_time_s
        # add features to block nodes in first layer only ?
        time_copy_s=time.time()
        for ntype in blocks[0].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                blocks[0].srcnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    blocks[0].srcnodes[ntype].data['_ID']]
        for ntype in blocks[-1].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                blocks[-1].dstnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    blocks[-1].dstnodes[ntype].data['_ID']]
            if g.nodes[ntype].data.get("h_clusters", None) is not None:
                blocks[-1].dstnodes[ntype].data['h_clusters']=g.nodes[ntype].data['h_clusters'][
                    blocks[-1].dstnodes[ntype].data['_ID']]
        for ntype in blocks[-1].ntypes:
            if g.nodes[ntype].data.get("motifs", None) is not None:
                blocks[-1].dstnodes[ntype].data['motifs']=g.nodes[ntype].data['motifs'][
                    blocks[-1].dstnodes[ntype].data['_ID']]
        time_copy=time.time()-time_copy_s
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)

        block_sample_time=time.time()-block_sample_s
        #print('copy time')
        #print(time_copy)
        #print('frontier calculation time')
        #print(frontier_time)
        #print('overal sampling time')
        #print(block_sample_time)
        return seeds, blocks

class PanRepSampler:
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
        for ntype in p_blocks[-1].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                p_blocks[-1].dstnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    p_blocks[-1].dstnodes[ntype].data['_ID']]
        for ntype in p_blocks[-1].ntypes:
            if g.nodes[ntype].data.get("motifs", None) is not None:
                p_blocks[-1].dstnodes[ntype].data['motifs']=g.nodes[ntype].data['motifs'][
                    p_blocks[-1].dstnodes[ntype].data['_ID']]
        time_copy=time.time()-cops
        block_sample_time=time.time()-block_sample_s
        '''
        print('copy time')
        print(time_copy)
        print('frontier calculation time')
        print(frontier_time)
        print('overal sampling time')
        print(block_sample_time)
        '''
        return (bsize, p_g, n_g, p_blocks, n_blocks)

def unmask_nodes(g,masked_node_types):
    for ntype in g.ntypes:
        if ntype not in masked_node_types:
                g.nodes[ntype].data['features']=g.nodes[ntype].data['masked_values']
    return g