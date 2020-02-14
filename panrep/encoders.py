from model import PanRepRGCN
from dgl.nn.pytorch import RelGraphConv
from functools import partial
import torch
import torch.nn.functional as F

#TODO SEE DGI code architecture and inspire similar context. add the dgi losses

class EncoderRGCN(PanRepRGCN):
    def create_features(self):
        features = torch.arange(self.num_nodes)
        if self.use_cuda:
            features = features.cuda()
        return features

    def build_input_layer(self):
        return RelGraphConv(self.inp_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)
    # TODO different layers may have different number of hidden units current implementation prevents
    def build_hidden_layer(self, idx):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout)
    def build_class_output_layer(self):
        return RelGraphConv(self.h_dim, self.out_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, dim=1),
                self_loop=self.use_self_loop)

    def build_reconstruct_output_layer(self):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=partial(F.softmax, dim=1),
                self_loop=self.use_self_loop)
    def build_output_layer(self):
        return self.build_reconstruct_output_layer()

