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
                 link_prediction_task=False,use_cuda=False):
        super(PanRepRGCNHetero, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_reconstruction_task=use_reconstruction_task
        self.use_infomax_task=use_infomax_task
        self.link_prediction_task=link_prediction_task
        self.loss_over_all_nodes=loss_over_all_nodes
        self.infomax=MutualInformationDiscriminator(n_hidden=h_dim)
        self.use_cuda = use_cuda
        self.encoder = encoder
        self.linkPredictor = LinkPredictor(out_dim=h_dim,etypes=etypes,ntype2id=ntype2id,use_cuda=use_cuda)
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