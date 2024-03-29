''' 
This file contains the ParticleNet model. This is taken (and modified) from the implementation in https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleNet.py
'''


''' ParticleNet

Paper: "ParticleNet: Jet Tagging via Particle Clouds" - https://arxiv.org/abs/1902.08570

Adapted from the DGCNN implementation in https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.
'''
import numpy as np
import torch
import torch.nn as nn
import time 

# Find the k nearest neighbors for each point. For the first layer we use the (η,φ) coordinates of the particles as input. For the next layers we use the output of the previous layer. 
def knn(x, k): # x: (batch_size, num_dims, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1), x) # x.transpose(2, 1): flips the last two dimensions
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) # (batch_size, num_points, num_points)
    
    # Find the indices of the k nearest neighbors
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # Initially we have x: (batch_size, num_dims, num_points) and idx: (batch_size, num_points, k) 

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    
    # We concatenate the neighbors' features with the original features
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList() # nn.ModuleList allows us to use the modules in a Python list 
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm: # perform batch normalization which normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation: 
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        # We add a shortcut (or skip) connection to the output of the last convolutional layer. Check if the input and output dimensions are the same. If not, perform a convolution to match the dimensions.
        if in_feat == out_feats[-1]: # to check if the input and output dimensions are the same. 
            self.sc = None # if they are the same, we don't need to perform a convolution in order to add a shortcut connection
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # We add a shortcut (or skip) connection to the output of the last convolutional layer
        if self.sc: # Runs if the input and output dimensions are not the same -> perform a convolution to match the dimensions
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:       # Input and output dimensions are the same -> no need to perform a convolution 
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):
    # the default parameters are the same as the original ParticleNet-Lite 
    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))], # Two EdgeConv layers with 7 neighbors each and (32, 32, 32) and (64, 64, 64) output channels respectively
                 fc_params=[(128, 0.1)],                             # One fully connected layer with 128 output channels and a dropout rate of 0.1
                 use_fusion=True,                                    
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,                                
                 for_segmentation=False,                            
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1] # the input dim of the edge conv block is the output dim of the previous edge conv block expect 
                                                                              # for the first edge conv block where the input dim is the input dim of the ParticleNet
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
        # If we have not defined a particular mask, the default one is: Mask all points that have a zero feature vector (in case of padding the input) 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The ParticleNet code wants the inputs to have the shape (njets, nfeatures, nparticles) but our convention is the shape (njets, nparticles, nfeatures)
        points = points.transpose(1, 2).to(device) 
        features = features.transpose(1, 2).to(device) 

        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (njets, 1, hadrons) 
            
        points *= mask
        features *= mask

        # Since we'll use the points to calculate the distance between the points, 
        # we need to shift the coordinates of the particles that are the result of zero-padding \
        # to infinity so that they don't affect the distance calculation. 
        coord_shift = (mask == 0) * 1e9 # shape (njets, 1, hadrons)
        
        if self.use_counts:
            counts = mask.float().sum(dim=-1) # mask contains boolean True=1 and False = 0. Summing over the last 
                                              # dimension gives the number of True values for each jet, 
                                              # i.e. the number of hadrons
            counts = torch.max(counts, torch.ones_like(counts))  # >=1
            # the shape is (njets, 1, 1)
        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features

        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift # for the first layers use points=(η,φ) for the kNN
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask
                
#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1) # shape (njets, num_classes) 

        return output

# Used to convolve the input features to ParticleNetTagger to another dimension before passing it to the ParticleNet. See the class ParticleNetTagger below.
class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)

# This is only useful if you want to add extra information related to the vertices. This is NOT discussed in the ParticleNet paper, but its probably used at the LHC (?).
# For our purposes, we can ignore this. Directly use the ParticleNet class above.   
# This is NOT the original ParticleNetTagger implementation, since I was playing around with it. For the original implementation, see the ParticleNet github repo.
class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims, # number of features for each particle (pf = particle flow)
                 sv_features_dims, # number of features for each secondary vertex (sv). Ignore for our purposes 
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))], # Default size for ParticleNet-Lite
                 fc_params=[(128, 0.1)],                             # Default size for ParticleNet-Lite
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)

        
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        
        # Call ParticleNet 
        self.pn = ParticleNet(input_dims=3, # if we use the self.pv_conv this is the output dimension of self.pv_conv. For now we feed the input directly to ParticleNet without convolving it to another dim 
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
    
        # For debugging in case of a gradient computation error
        torch.autograd.set_detect_anomaly(True)

        # The code contains a lot of additions wrt the OG paper, e.g. sv features.
        # For simplicity we pass a zero vector for the sv features and a mask = None is passed to the ParticleNet.
        # In the next class (Particle Net) the mask is set to the default mask (mask all points that have a zero feature vector)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transpose the input to the correct shape (njets, feature_dim, hadrons) and move it to gpu if available
        pf_points = pf_points.transpose(1, 2).to(device) 
        pf_features = pf_features.transpose(1, 2).to(device)
        sv_points = sv_points.transpose(1, 2).to(device)
        sv_features = sv_features.transpose(1, 2).to(device)
        pf_mask = pf_mask.transpose(1, 2).to(device)
        sv_mask = sv_mask.transpose(1, 2).to(device)

        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask
            
        #points = torch.cat((pf_points, sv_points), dim=2)
        #features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        points = pf_points
        features = pf_features
        mask = pf_mask
        
        return self.pn(points, features, mask = None)   