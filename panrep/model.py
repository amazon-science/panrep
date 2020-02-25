import torch.nn as nn
import torch as th
import dgl.function as fn
from encoders import EncoderRGCN,EncoderRelGraphConvHetero
from node_supervision_tasks import AttributeDecoder,MultipleAttributeDecoder,MutualInformationDiscriminator
import torch
class PanRepRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, inp_dim,out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0, reconstruct_dim=1,
                 use_self_loop=False, use_cuda=False):
        super(PanRepRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.inp_dim=inp_dim
        self.out_dim = out_dim
        self.reconstruct_dim=reconstruct_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.encoder = EncoderRGCN(num_nodes, h_dim, inp_dim, out_dim, num_rels, num_bases,
                                 num_hidden_layers, dropout, use_self_loop, use_cuda)
        # create rgcn layers
        #self.encoder.build_model()

        self.attributeDecoder = AttributeDecoder(self.h_dim,self.reconstruct_dim)

    def forward(self, g, h, r, norm):
        h=self.encoder(g,h,r,norm)
        #for layer in self.layers:
        #    h = layer(g, h, r, norm)
        # TODO write more neatly this layer performs attribute reconstruction
        reconstructed=self.attributeDecoder(h)

        return reconstructed.squeeze(),h




class PanRepRGCNHetero(nn.Module):
    def __init__(self, g,
                 h_dim,  out_dim,
                 encoder,
                 masked_node_types=[],
                 num_hidden_layers=1,
                 dropout=0,loss_over_all_nodes=False, use_reconstruction_loss=True, use_infomax_loss=True):
        super(PanRepRGCNHetero, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_reconstruction_loss=use_reconstruction_loss
        self.use_infomax_loss=use_infomax_loss
        self.loss_over_all_nodes=loss_over_all_nodes
        self.G=g
        self.infomax=MutualInformationDiscriminator(n_hidden=h_dim)
        #self.use_cuda = use_cuda
        self.encoder = encoder

        # create rgcn layers
        # self.encoder.build_model()
        # G.nodes['transaction'].data['features']
        self.out_size_dict = {};
        if not self.use_infomax_loss and not self.use_reconstruction_loss:
            raise ValueError("All losses disabled, can not train.")
        for name in self.G.ntypes:
            self.out_size_dict[name] = self.G.nodes[name].data['features'].size(1);
        self.attributeDecoder = MultipleAttributeDecoder(
            out_size_dict=self.out_size_dict, in_size=self.h_dim, h_dim=h_dim, G=self.G,masked_node_types=masked_node_types,
        loss_over_all_nodes=loss_over_all_nodes)
    def updated_graph(self,g):
        self.G=g
        self.encoder.G=g
        self.attributeDecoder.G=g

    def forward(self,masked_nodes):

        #h=self.encoder(corrupt=False)
        positive = self.encoder(corrupt=False)
        loss=0
        if self.use_infomax_loss:
            negative = self.encoder(corrupt=True)
            infomax_loss = self.infomax(positive, negative)
            loss += infomax_loss
        if self.use_reconstruction_loss:
            reconstruct_loss = self.attributeDecoder(positive,masked_nodes=masked_nodes)
            loss += reconstruct_loss

        return loss, positive