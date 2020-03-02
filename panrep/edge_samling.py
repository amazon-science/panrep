import torch
import numpy as np

def create_edge_mask(old_g):
    g=old_g.local_var()
    for etype in g.etypes:
        g.edges[etype].data['mask']=torch.tensor(np.ones((g.number_of_edges(etype))))
    return g

def generate_neg_samples(u,v,sampl_ids):
    return
def edge_sampler(old_g,num_sampled_edges,negative_rate):
    pos_samples={}
    neg_samples={}
    g = old_g.local_var()
    for etype in g.etypes:
        pos_samples[etype]=[]
        u,v,eid=g.all_edges(form='all', etype=etype)
        sampl_ids=np.random.choice(g.number_of_edges(etype),size=num_sampled_edges,replace=False)
        pos_samples[etype]=(u[sampl_ids],v[sampl_ids])
        masked_edges=eid[sampl_ids]
        g.edges[etype].data['mask'][masked_edges] = 0
        # TODO negative samples
        # neg_samples[etype]=generate_neg_samples(u,v,sampl_ids,negative_rate)


    # create function consuming pos
    return g,pos_samples,neg_samples