import torch.nn as nn
import torch as th
import dgl.function as fn
from encoders import EncoderRGCN,EncoderRelGraphConvHetero
from node_supervision_tasks import AttributeDecoder,MultipleAttributeDecoder\
    ,MutualInformationDiscriminator,LinkPredictor

class PanRepRGCNHetero(nn.Module):
    def __init__(self,
                 h_dim, out_dim,
                 encoder,
                 in_size_dict,
                 etypes,
                 ntype2id,
                 masked_node_types=[],
                 num_hidden_layers=1,
                 dropout=0, loss_over_all_nodes=False, use_reconstruction_task=True, use_infomax_task=True,
                 link_prediction_task=False,use_cuda=False,average_across_node_types=False,link_predictor=None):
        super(PanRepRGCNHetero, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_reconstruction_task=use_reconstruction_task
        self.use_infomax_task=use_infomax_task
        self.link_prediction_task=link_prediction_task
        self.loss_over_all_nodes=loss_over_all_nodes
        self.infomax=MutualInformationDiscriminator(n_hidden=h_dim,average_across_node_types=average_across_node_types)
        self.use_cuda = use_cuda
        self.encoder = encoder
        if link_predictor is None:
            self.linkPredictor = LinkPredictor(out_dim=h_dim,etypes=etypes,ntype2id=ntype2id,use_cuda=use_cuda)
        else:
            self.linkPredictor=link_predictor
        # create rgcn layers
        # self.encoder.build_model()
        # G.nodes['transaction'].data['features']
        self.out_size_dict = {};
        if not self.use_infomax_task and not self.use_reconstruction_task and not self.link_prediction_task:
            raise ValueError("All losses disabled, can not train.")
        for name in in_size_dict.keys():
            self.out_size_dict[name] =in_size_dict[name]
        self.attributeDecoder = MultipleAttributeDecoder(
            out_size_dict=self.out_size_dict, in_size=self.h_dim, h_dim=h_dim,masked_node_types=masked_node_types,
        loss_over_all_nodes=loss_over_all_nodes)

    def forward(self, g, masked_nodes, sampled_links, sampled_link_labels):

        #h=self.encoder(corrupt=False)
        positive = self.encoder(g,corrupt=False)
        loss=0

        if self.use_infomax_task:
            negative = self.encoder(g,corrupt=True)
            infomax_loss = self.infomax(positive, negative)
            loss += infomax_loss

        if self.use_reconstruction_task:
            reconstruct_loss = self.attributeDecoder(g,positive,masked_nodes=masked_nodes)
            loss += reconstruct_loss

        if self.link_prediction_task:
            link_prediction_loss=self.linkPredictor(g,positive, sampled_links, sampled_link_labels)
            loss += link_prediction_loss

        return loss, positive

    def forward_mb(self, p_blocks, perm_emb, masked_nodes, p_g=None, n_g=None, n_blocks=None,
                   num_chunks=None, chunk_size=None, neg_sample_size=None):

        #h=self.encoder(corrupt=False)
        positive = self.encoder.forward_mb(p_blocks)
        loss=0
        reconstruct_loss=0
        infomax_loss=0
        link_prediction_loss=0
        if self.use_infomax_task:
            # The following assings the permuted attributes as the new graph features
            # TODO ensure that the original node features are not accessed again.
            for ntype in p_blocks[0].ntypes:
                p_blocks[0].nodes[ntype].data['h_f'] = perm_emb[ntype]

            negative_infomax = self.encoder.forward_mb(p_blocks)
            infomax_loss = self.infomax.forward_mb(positive, negative_infomax)
            loss += infomax_loss
            print("Infomax loss {}".format(
            infomax_loss.detach()))

        if self.use_reconstruction_task:
            reconstruct_loss = self.attributeDecoder.forward_mb(p_blocks[-1], positive, masked_nodes=masked_nodes)
            loss += reconstruct_loss
            print("Reconstruct loss {}".format(
            reconstruct_loss.detach()))

        if self.link_prediction_task:
            negative_link_prediction = self.encoder.forward_mb(n_blocks)
            link_prediction_loss=self.linkPredictor.get_loss(positive, p_g, negative_link_prediction,n_g,
                                  num_chunks, chunk_size, neg_sample_size)
            loss += link_prediction_loss
            print("Link prediction loss:{}".format(
            link_prediction_loss.detach()))

        return loss, positive
