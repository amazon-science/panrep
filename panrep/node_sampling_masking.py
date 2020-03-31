import random
import copy
import torch
import numpy as np
import dgl
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
        return seeds, blocks


class PanRepNeighborSampler:
    def __init__(self, g, fanouts, device, full_neighbor=False):
        """
        if fanouts is None, sample full neighbor
        """
        self.g = g
        self.fanouts = fanouts
        self.device=device
        self.full_neighbor = full_neighbor
        #self.pct_batch=pct_batch


        counter = itertools.count(0)
        self.hetero_map = {next(counter):(i,ntype)    for ntype in (g.ntypes) for i in range(g.number_of_nodes(ntype))}
        self.number_of_nodes=len(self.hetero_map)

    def sample_blocks(self, seeds_list):
        blocks = []
        seeds={}
        g=self.g
        device=self.device
        seeds_list.sort()
        for s in seeds_list:
            nid,ntype=self.hetero_map[s]
            if ntype not in seeds:
                seeds[ntype]=[nid]
            else:
                seeds[ntype]+=[nid]
        for ntype in seeds:
            seeds[ntype]=torch.tensor(seeds[ntype])#.to(device)
        cur = seeds
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
        # add features to block nodes in first layer only ?
        for ntype in blocks[0].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                blocks[0].srcnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    blocks[0].srcnodes[ntype].data['_ID']]
        for ntype in blocks[-1].ntypes:
            if g.nodes[ntype].data.get("h_f", None) is not None:
                blocks[-1].dstnodes[ntype].data['h_f']=g.nodes[ntype].data['h_f'][
                    blocks[-1].dstnodes[ntype].data['_ID']]
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(device)
        return seeds, blocks

def unmask_nodes(g,masked_node_types):
    for ntype in g.ntypes:
        if ntype not in masked_node_types:
                g.nodes[ntype].data['features']=g.nodes[ntype].data['masked_values']
    return g