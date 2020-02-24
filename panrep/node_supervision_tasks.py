import torch.nn.functional as F
import torch.nn as nn
import torch
import math

def graph_link_prediction():
    return

class LinkPredictor(nn.Module):
    def __init__(self, out_dim, num_rels, reg_param=0):
        super(LinkPredictor, self).__init__()
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, out_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self,  embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

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
    def __init__(self, out_size_dict, in_size, h_dim, G,masked_node_types, loss_over_all_nodes,activation=nn.ReLU()):
        '''

        :param out_size_dict:
        :param in_size:
        :param h_dim:
        :param G:
        :param masked_node_types: Node types to not penalize for reconstruction
        :param activation:
        '''

        super(MultipleAttributeDecoder, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim
        self.G=G
        self.weight=nn.ModuleDict()
        self.masked_node_types=masked_node_types
        self.loss_over_all_nodes=loss_over_all_nodes

        for name in self.G.ntypes:
            layers=[]
            layers.append(nn.Linear( in_size, self.h_dim))
            layers.append(activation)
            layers.append(nn.Linear( self.h_dim, out_size_dict[name],bias=False))
            self.weight[name]=nn.Sequential(*layers)

        #self.weight = nn.ModuleDict({
        #    name: nn.Linear( in_size, out_size_dict[name],bias=False) for name in self.G.ntypes
        #})

    def forward(self,h,masked_nodes):
        g = self.G.local_var()
        loss=0
        for i, ntype in enumerate(g.ntypes):
            g.nodes[ntype].data['x'] = h[i]
        for name in self.weight:
            if name not in self.masked_node_types:
                g.nodes[name].data['h']=self.weight[name](g.nodes[name].data['x'])
                if bool(masked_nodes):
                    if not self.loss_over_all_nodes:
                        loss+=F.mse_loss(g.nodes[name].data['h'][masked_nodes[name]],g.nodes[name].data['masked_values'][masked_nodes[name]])
                    else:
                        loss += F.mse_loss(g.nodes[name].data['h'],
                                           g.nodes[name].data['masked_values'])

                else:
                    loss += F.mse_loss(g.nodes[name].data['h'],
                                       g.nodes[name].data['features'])
        #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]

        return loss


def reconstruction_loss(reconstructed_feats, feats):
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