import torch.nn.functional as F

def node_attribute_reconstruction(reconstructed_feats,feats):
    feats=feats.float()
    loss_train = F.mse_loss(reconstructed_feats, feats)

    return loss_train
