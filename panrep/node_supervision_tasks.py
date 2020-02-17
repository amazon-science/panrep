import torch.nn.functional as F
import torch.nn as nn
import torch
import math
class AttributeDecoder(nn.Module):
    def __init__(self, h_dim, reconstruct_dim=1, use_cuda=False):
        super(AttributeDecoder, self).__init__()
        self.h_dim=h_dim
        self.reconstruct_dim=reconstruct_dim
        self.reconstruction_layer = nn.Linear(self.h_dim, self.reconstruct_dim)
    def forward(self, h):
        h=self.reconstruction_layer(h)
        return h
class MultipleAttributeDecoder(nn.Module):
    def __init__(self, out_size_dict, in_size, h_dim, G, activation=nn.ReLU()):
        super(MultipleAttributeDecoder, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim
        self.G=G
        self.weight=nn.ModuleDict()
        for name in self.G.ntypes:
            layers=[]
            layers.append(nn.Linear( in_size, self.h_dim))
            layers.append(activation)
            layers.append(nn.Linear( self.h_dim, out_size_dict[name],bias=False))
            self.weight[name]=nn.Sequential(*layers)

        #self.weight = nn.ModuleDict({
        #    name: nn.Linear( in_size, out_size_dict[name],bias=False) for name in self.G.ntypes
        #})

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

class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features

class MutualInformationDiscriminator(nn.Module):
    # returns the MI loss function follows the dgl implementation
    def __init__(self, n_hidden):
        super(MutualInformationDiscriminator, self).__init__()
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, positives, negatives):
        l1=0
        l2=0
        for positive,negative in zip(positives,negatives):
            summary = torch.sigmoid(positive.mean(dim=0))

            positive = self.discriminator(positive, summary)
            negative = self.discriminator(negative, summary)

            l1 += self.loss(positive, torch.ones_like(positive))
            l2 += self.loss(negative, torch.zeros_like(negative))
        return l1+l2