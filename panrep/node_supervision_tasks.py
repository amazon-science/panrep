import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class LinkPredictorDistMultMB(nn.Module):
    def __init__(self,
                 g,
                 device,
                 h_dim,
                 num_rels,
                 dropout=0,
                 use_self_loop=False,
                 regularization_coef=0,etype_key_map=None):
        super(LinkPredictorDistMultMB, self).__init__()
        self.g = g
        self.device = device
        self.h_dim = h_dim
        self.etype_key_map=etype_key_map
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.regularization_coef = regularization_coef

        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim).to(self.device))
        nn.init.xavier_uniform_(self.w_relation)



    def regularization_loss(self, h_emb, t_emb, nh_emb, nt_emb):
        return torch.mean(h_emb.pow(2)) + \
               torch.mean(t_emb.pow(2)) + \
               torch.mean(nh_emb.pow(2)) + \
               torch.mean(nt_emb.pow(2)) + \
               torch.mean(self.w_relation.pow(2))

    def calc_pos_score(self, h_emb, t_emb, rids):
        # DistMult
        r = self.w_relation[rids]
        score = torch.sum(h_emb * r * t_emb, dim=-1)
        return score

    def calc_neg_tail_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = heads.shape[1]
        r = self.w_relation[rids]
        tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
        tails = torch.transpose(tails, 1, 2)
        tmp = (heads * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, tails)

    def calc_neg_head_score(self, heads, tails, rids, num_chunks, chunk_size, neg_sample_size):
        hidden_dim = tails.shape[1]
        r = self.w_relation[rids]
        heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
        heads = torch.transpose(heads, 1, 2)
        tmp = (tails * r).reshape(num_chunks, chunk_size, hidden_dim)
        return torch.bmm(tmp, heads)
    def get_loss(self,p_h, p_g, n_h,n_g, num_chunks, chunk_size, neg_sample_size):
        # loss calculation
        for ntype, emb in p_h.items():
            p_g.nodes[ntype].data['h'] = emb
        for ntype, emb in n_h.items():
            n_g.nodes[ntype].data['h'] = emb
        p_head_emb = []
        p_tail_emb = []
        rids = []
        for canonical_etype in p_g.canonical_etypes:
            head, tail = p_g.all_edges(etype=canonical_etype)
            head_emb = p_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = p_g.nodes[canonical_etype[2]].data['h'][tail]
            idx = self.etype_key_map[canonical_etype]
            rids.append(torch.full((head_emb.shape[0],), idx, dtype=torch.long))
            p_head_emb.append(head_emb)
            p_tail_emb.append(tail_emb)
        n_head_emb = []
        n_tail_emb = []
        for canonical_etype in n_g.canonical_etypes:
            head, tail = n_g.all_edges(etype=canonical_etype)
            head_emb = n_g.nodes[canonical_etype[0]].data['h'][head]
            tail_emb = n_g.nodes[canonical_etype[2]].data['h'][tail]
            n_head_emb.append(head_emb)
            n_tail_emb.append(tail_emb)
        p_head_emb = torch.cat(p_head_emb, dim=0)
        p_tail_emb = torch.cat(p_tail_emb, dim=0)
        rids = torch.cat(rids, dim=0)
        n_head_emb = torch.cat(n_head_emb, dim=0)
        n_tail_emb = torch.cat(n_tail_emb, dim=0)
        assert p_head_emb.shape[0] == p_tail_emb.shape[0]
        assert rids.shape[0] == p_head_emb.shape[0]
        assert n_head_emb.shape[0] == n_tail_emb.shape[0]
        n_shuffle_seed = torch.randperm(n_head_emb.shape[0])
        n_head_emb = n_head_emb[n_shuffle_seed]
        n_tail_emb = n_tail_emb[n_shuffle_seed]

        loss = self.get_loss_h(p_head_emb,
                              p_tail_emb,
                              n_head_emb,
                              n_tail_emb,
                              rids,
                              num_chunks,
                              chunk_size,
                              neg_sample_size)
        return loss

    def get_loss_h(self, h_emb, t_emb, nh_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        pos_score = self.calc_pos_score(h_emb, t_emb, rids)
        t_neg_score = self.calc_neg_tail_score(h_emb, nt_emb, rids, num_chunks, chunk_size, neg_sample_size)
        h_neg_score = self.calc_neg_head_score(nh_emb, t_emb, rids, num_chunks, chunk_size, neg_sample_size)
        pos_score = F.logsigmoid(pos_score)
        h_neg_score = h_neg_score.reshape(-1, neg_sample_size)
        t_neg_score = t_neg_score.reshape(-1, neg_sample_size)
        h_neg_score = F.logsigmoid(-h_neg_score).mean(dim=1)
        t_neg_score = F.logsigmoid(-t_neg_score).mean(dim=1)

        pos_score = pos_score.mean()
        h_neg_score = h_neg_score.mean()
        t_neg_score = t_neg_score.mean()
        predict_loss = -(2 * pos_score + h_neg_score + t_neg_score)

        reg_loss = self.regularization_loss(h_emb, t_emb, nh_emb, nt_emb)

        print("pos loss {}, neg loss {}|{}, reg_loss {}".format(pos_score.detach(),
                                                                h_neg_score.detach(),
                                                                t_neg_score.detach(),
                                                                self.regularization_coef * reg_loss.detach()))
        return predict_loss + self.regularization_coef * reg_loss

class LinkPredictor(nn.Module):
    def __init__(self, out_dim, etypes, ntype2id,reg_param=0,use_cuda=False):
        super(LinkPredictor, self).__init__()
        self.reg_param = reg_param
        self.etypes=etypes
        self.w_relation={}
        self.ntype2id=ntype2id
        for ename in self.etypes:
            if use_cuda:
                self.w_relation[ename] = nn.Parameter(torch.Tensor(out_dim,1)).cuda()
            else:
                self.w_relation[ename] = nn.Parameter(torch.Tensor(out_dim,1))
            nn.init.xavier_uniform_(self.w_relation[ename],
                                gain=nn.init.calculate_gain('relu'))


    def calc_score(self, g,embedding, dict_s_d):
        # DistMult
        score={}

        for etype in self.etypes:
            (stype,e,dtype)=g.to_canonical_etype(etype)
            s = embedding[self.ntype2id[stype]][dict_s_d[etype][:, 0]]
            r = self.w_relation[etype].squeeze()
            o = embedding[self.ntype2id[dtype]][dict_s_d[etype][:, 1]]
            score[etype] = torch.sum(s * r * o, dim=1)
        return score

    def regularization_loss(self, embedding):
            loss=0
            for e in embedding:
                loss+=torch.mean(e.pow(2))

            for e in self.w_relation.keys():
                loss+=torch.mean(self.w_relation[e].pow(2))
            return loss
    def forward(self, g,embed, edict_s_d, e_dict_labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(g,embed, edict_s_d)
        predict_loss=0
        for etype in self.etypes:
            predict_loss += F.binary_cross_entropy_with_logits(score[etype], e_dict_labels[etype])

        # TODO implement regularization

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

class NodeMotifDecoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dict, distribution=False,activation=nn.ReLU(), single_layer=False):
        '''

        :param out_dim:
        :param in_dim:
        :param h_dim:
        :param activation:
        '''

        super(NodeMotifDecoder, self).__init__()
        self.activation=activation
        self.h_dim=h_dim
        self.weight=nn.ModuleDict()
        self.single_layer=single_layer
        self.distribution=distribution
        layers=[]
        for name in out_dict.keys():
            layers=[]
            if self.single_layer:
                layers.append(nn.Linear( in_dim,  out_dict[name],bias=False))
            else:
                layers.append(nn.Linear( in_dim, self.h_dim))
                layers.append(activation)
                layers.append(nn.Linear( self.h_dim, out_dict[name],bias=False))
            self.weight[name]=nn.Sequential(*layers)

    def forward(self, g, h):
        node_embed=h
        loss=0
        g = g.local_var()
        for name in g.dsttypes:
            if name in node_embed:
                reconstructed=self.weight[name](node_embed[name])
                if self.distribution:
                    loss += F.mse_loss(reconstructed, g.dstnodes[name].data['motifs'])
                else:
                    loss += F.binary_cross_entropy_with_logits(reconstructed, g.dstnodes[name].data['motifs'])
            #hs = [g.nodes[ntype].data['h'] for ntype in g.ntypes]

        return loss


class MultipleAttributeDecoder(nn.Module):
    def __init__(self, out_size_dict, in_size, h_dim, masked_node_types, loss_over_all_nodes,activation=nn.ReLU(),single_layer=False):
        '''

        :param out_size_dict:
        :param in_size:
        :param h_dim:
        :param masked_node_types: Node types to not penalize for reconstruction
        :param activation:
        '''

        super(MultipleAttributeDecoder, self).__init__()
        # W_r for each node
        self.activation=activation
        self.h_dim=h_dim
        self.weight=nn.ModuleDict()
        self.masked_node_types=masked_node_types
        self.loss_over_all_nodes=loss_over_all_nodes
        self.single_layer=single_layer

        for name in out_size_dict.keys():
            layers=[]
            if self.single_layer:
                layers.append(nn.Linear( in_size,  out_size_dict[name],bias=False))
            else:
                layers.append(nn.Linear( in_size, self.h_dim))
                layers.append(activation)
                layers.append(nn.Linear( self.h_dim, out_size_dict[name],bias=False))
            self.weight[name]=nn.Sequential(*layers)


    def forward(self,G,h,masked_nodes):
        g = G.local_var()
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

    def forward_mb(self, g, h, masked_nodes):
        node_embed=h
        loss=0
        g = g.local_var()
        for name in self.weight:
            if g.dstnodes[name].data.get("h_f", None) is not None:
                reconstructed=self.weight[name](node_embed[name])
                if bool(masked_nodes):
                    if not self.loss_over_all_nodes:
                        loss+=F.mse_loss(reconstructed[masked_nodes[name]],g.dstnodes[name].data['h_f'])
                    else:
                        loss += F.mse_loss(reconstructed, g.dstnodes[name].data['h_f'])

                else:
                    loss += F.mse_loss(reconstructed, g.dstnodes[name].data['h_f'])
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
    def __init__(self, n_hidden,average_across_node_types=False):
        super(MutualInformationDiscriminator, self).__init__()
        self.discriminator = Discriminator(n_hidden)
        self.loss = nn.BCEWithLogitsLoss()
        self.average_across_node_types=average_across_node_types
        # keep a global summary
        #self.positives


    def forward(self, positives, negatives):
        l1=0
        l2=0
        if self.average_across_node_types:
            positives=torch.cat(positives,dim=0)
            negatives=torch.cat(negatives,dim=0)
            summary = torch.sigmoid(positives.mean(dim=0))
            positive = self.discriminator(positives.mean(dim=0), summary)
            negative = self.discriminator(negatives.mean(dim=0), summary)
            l1 += self.loss(positive, torch.ones_like(positive))
            l2 += self.loss(negative, torch.zeros_like(negative))

            return l1 + l2
        else:
            for positive,negative in zip(positives,negatives):
                summary = torch.sigmoid(positive.mean(dim=0))

                positive = self.discriminator(positive, summary)
                negative = self.discriminator(negative, summary)

                l1 += self.loss(positive, torch.ones_like(positive))
                l2 += self.loss(negative, torch.zeros_like(negative))
            return l1+l2

    def forward_mb(self, positives, negatives):
        l1=0
        l2=0
        # TODO summary per node type or across all node types? for infomax

        for ntype in positives.keys():
                positive=positives[ntype]
                negative=negatives[ntype]
                summary = torch.sigmoid(positive.mean(dim=0))

                positive = self.discriminator(positive, summary)
                negative = self.discriminator(negative, summary)

                l1 += self.loss(positive, torch.ones_like(positive))
                l2 += self.loss(negative, torch.zeros_like(negative))
        return l1+l2