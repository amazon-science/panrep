import torch.nn.functional as F

def node_attribute_reconstruction(reconstructed_feats,feats):
    loss_train = F.mse_loss(reconstructed_feats, feats)
    return loss_train
