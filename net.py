import torch
import torch.nn as nn
import sys
sys.path.append('/home/chenxu/Downloads/TRI_chen/diffusion_policy_orig')
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

#### On O_t and A_t. Smaller but same policy network
def get_unet(input_dim, global_cond_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    )

class RNDPolicy(nn.Module):
    def __init__(self, input_dim, global_cond_dim):
        super(RNDPolicy, self).__init__()
        self.predictor = get_unet(input_dim, global_cond_dim)

        self.target = get_unet(input_dim, global_cond_dim)

        for param in self.target.parameters():
            param.requires_grad = False
            
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, action, observation):
        t = torch.zeros(len(action)).to(action.device)
        target_feature = self.target(action, t, global_cond=observation)
        predict_feature = self.predictor(action, t, global_cond=observation)
        predict_error = self.dist(predict_feature, target_feature)
        return predict_error
  
  
#### On O_t only    
def get_unet2(input_dim):
    return ConditionalUnet1D(
        input_dim=input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=128,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=False
    )

class RNDPolicy2(nn.Module):
    def __init__(self, input_dim):
        super(RNDPolicy2, self).__init__()
        self.predictor = get_unet2(input_dim)

        self.target = get_unet2(input_dim)

        for param in self.target.parameters():
            param.requires_grad = False
            
        self.dist = nn.PairwiseDistance(p=2)

    def forward(self, observation):
        target_feature = self.target(observation, 0)
        predict_feature = self.predictor(observation, 0)
        predict_error = self.dist(predict_feature, target_feature)
        return predict_error