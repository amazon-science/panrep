import torch.nn as nn
import torch as th
import dgl.function as fn
from encoders import EncoderRGCN,EncoderRelGraphConvHetero
from node_supervision_tasks import NodeMotifDecoder,MultipleAttributeDecoder\
    ,MutualInformationDiscriminator,LinkPredictor

class PanRepHetero(nn.Module):
    def __init__(self,
                 h_dim, out_dim,
                 encoder,
                 out_size_dict,
                 etypes,
                 ntype2id,
                 masked_node_types=[],
                 num_hidden_layers=1,
                 dropout=0, loss_over_all_nodes=False, use_reconstruction_task=True, use_infomax_task=True,
                 link_prediction_task=False,use_cuda=False,average_across_node_types=False,
                 use_node_motif=False,link_predictor=None,out_motif_dict=None, use_cluster=False,
                 single_layer=False,classifier=None,metapathRWSupervision=None,focus_category=False):
        super(PanRepHetero, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_reconstruction_task=use_reconstruction_task
        self.use_infomax_task=use_infomax_task
        self.link_prediction_task=link_prediction_task
        self.loss_over_all_nodes=loss_over_all_nodes
        self.use_node_motif_task=use_node_motif
        self.focus_category=focus_category
        self.classifier=classifier
        if metapathRWSupervision is not None:
            self.rw_supervision_task = True
        else:
            self.rw_supervision_task = False
        self.metapathRWSupervision=metapathRWSupervision

        self.infomax=MutualInformationDiscriminator(n_hidden=h_dim,average_across_node_types=average_across_node_types,focus_category=focus_category)
        self.use_cuda = use_cuda
        self.encoder = encoder
        if link_predictor is None:
            self.link_prediction_task = False
        else:
            self.link_prediction_task=True
            self.linkPredictor=link_predictor
        # create rgcn layers
        # self.encoder.build_model()
        # G.nodes['transaction'].data['features']

        #if not self.use_infomax_task and not self.use_reconstruction_task \
        #        and not self.link_prediction_task and not self.use_node_motif_task:
        #    raise ValueError("All losses disabled, can not train.")
        if self.use_reconstruction_task:
            self.attributeDecoder = MultipleAttributeDecoder(
                out_size_dict=out_size_dict, in_size=self.h_dim,
                h_dim=h_dim,masked_node_types=masked_node_types,
            loss_over_all_nodes=loss_over_all_nodes,single_layer=single_layer,use_cluster=use_cluster)
        if self.use_node_motif_task:
            self.nodeMotifDecoder=NodeMotifDecoder(in_dim=self.h_dim, h_dim=self.h_dim, out_dict=out_motif_dict)

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
        if self.use_node_motif_task:
            motif_loss = self.nodeMotifDecoder(g,positive)
            loss += motif_loss
        if self.link_prediction_task:
            link_prediction_loss=self.linkPredictor(g,positive, sampled_links, sampled_link_labels)
            loss += link_prediction_loss

        return loss, positive
    def classifier_forward_mb(self,p_blocks):
        encoding=self.encoder.forward_mb(p_blocks)
        logits={}
        for ntype in encoding.keys():
            logits[ntype]=self.classifier.forward(encoding[ntype])
        return logits
    def link_predictor_forward_mb(self,p_blocks):
        encoding=self.encoder.forward_mb(p_blocks)
        link_prediction_loss = self.linkPredictor.forward_mb(g=p_blocks[-1], embed=encoding)
        link_prediction_loss
        print("Link prediction loss:{}".format(
            link_prediction_loss.detach()))
        return link_prediction_loss
    def classifier_forward(self,g):
        encoding=self.encoder.forward(g)
        logits={}
        for ntype in encoding.keys():
            logits[ntype]=self.classifier.forward(encoding[ntype])
        return logits

    def forward_mb(self, p_blocks, masked_nodes=None, rw_neighbors=None):

        #h=self.encoder(corrupt=False)
        positive = self.encoder.forward_mb(p_blocks)
        loss=0
        if self.use_infomax_task:
            negative_infomax = self.encoder.forward_mb(p_blocks,permute=True)
            infomax_loss = self.infomax.forward_mb(positive, negative_infomax)
            loss += infomax_loss
            print("Infomax loss {}".format(
            infomax_loss.detach()))

        if self.use_reconstruction_task:
            reconstruct_loss = self.attributeDecoder.forward_mb(p_blocks[-1], positive, masked_nodes=masked_nodes)
            loss += reconstruct_loss
            print("Reconstruct loss {}".format(
            reconstruct_loss.detach()))
        if self.use_node_motif_task:
            motif_loss = self.nodeMotifDecoder(p_blocks[-1],positive)
            loss += motif_loss
            print("Node motif loss {}".format(
            motif_loss.detach()))
        if self.link_prediction_task:
            link_prediction_loss=self.linkPredictor.forward_mb(g=p_blocks[-1], embed=positive)
            loss += link_prediction_loss
            print("Link prediction loss:{}".format(
            link_prediction_loss.detach()))
        if self.rw_supervision_task and rw_neighbors is not None:
           meta_loss=self.metapathRWSupervision.get_loss(g=p_blocks[-1], embed=positive,
                                           rw_neighbors=rw_neighbors)
           loss += meta_loss
           print("meta_loss: {:.4f}".format(meta_loss.item()))

        return loss, positive
