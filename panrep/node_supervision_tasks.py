import torch.nn.functional as F
import torch.nn as nn
import torch

class AttributeDecoder(nn.Module):
    def __init__(self, h_dim, reconstruct_dim=1, use_cuda=False):
        super(AttributeDecoder, self).__init__()
        self.h_dim=h_dim
        self.reconstruct_dim=reconstruct_dim
        self.reconstruction_layer = nn.Linear(self.h_dim, self.reconstruct_dim)
    def forward(self, h):
        h=self.reconstruction_layer(h)
        return h
class HeteroRGCNLayerFirst(nn.Module):
    def __init__(self, in_size_dict, out_size, ntypes):
        super(HeteroRGCNLayerFirst, self).__init__()
        # W_r for each node
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size_dict[name], out_size, bias=False) for name in ntypes
        })

    def forward(self, G):
        for name in self.weight:
            G.apply_nodes(lambda nodes: {'h': self.weight[name](nodes.data['features'])}, ntype=name);
        hs = [G.nodes[ntype].data['h'] for ntype in G.ntypes]
        return hs
class MultipleAttributeDecoder(nn.Module):
    def __init__(self, out_size_dict, in_size, G):
        super(MultipleAttributeDecoder, self).__init__()
        # W_r for each node
        self.G=G
        self.weight = nn.ModuleDict({
            name: nn.Linear( in_size, out_size_dict[name],bias=False) for name in self.G.ntypes
        })
    def forward(self,h):
        g = self.G.local_var()
        loss=0
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = h[i]
        for name in self.weight:
            g.nodes[name].data['h']=self.weight[name](g.nodes[name].data['x'])
            loss+=F.mse_loss(g.nodes[name].data['h'],g.nodes[name].data['features'])
        #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]

        return loss


def node_attribute_reconstruction(reconstructed_feats,feats):
    feats=feats.float()
    loss_train = F.mse_loss(reconstructed_feats, feats)

    return loss_train


class MutualInformationDiscriminator(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(MutualInformationDiscriminator, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features