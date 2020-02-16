import torch.nn as nn
import torch as th
import dgl.function as fn
from encoders import EncoderRGCN,EncoderRelGraphConvHetero
from node_supervision_tasks import AttributeDecoder,MultipleAttributeDecoder

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
                 num_bases,
                 num_hidden_layers=1,
                 reconstruct_dim=1,
                 dropout=0,
                 use_self_loop=False):
        super(PanRepRGCNHetero, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.reconstruct_dim=reconstruct_dim
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.num_bases=num_bases
        self.use_self_loop = use_self_loop
        self.G=g
        #self.use_cuda = use_cuda
        self.encoder = EncoderRelGraphConvHetero(self.G,
                 self.h_dim, self.out_dim,
                 self.num_bases,
                 self.num_hidden_layers,
                 self.dropout,
                 self.use_self_loop)
        # create rgcn layers
        # self.encoder.build_model()
        # G.nodes['transaction'].data['features']
        self.out_size_dict = {};
        for name in self.G.ntypes:
            self.out_size_dict[name] = self.G.nodes[name].data['features'].size(1);
        self.attributeDecoder = MultipleAttributeDecoder(
            out_size_dict=self.out_size_dict, in_size=self.h_dim, G=self.G)

    def forward(self):
        h=self.encoder()
        # TODO write more neatly this layer performs attribute reconstruction
        reconstruct_loss=self.attributeDecoder(h)

        return reconstruct_loss,h