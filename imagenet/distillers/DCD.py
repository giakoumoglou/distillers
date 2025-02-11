# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCDLoss(nn.Module):
    """
    Discriminative and Consistent Knowledge Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        max_tau (float): Maximum clamp value for the temperature scaling tau.
        alpha (float): Scaling factor for the KL-divergence part of the loss.
        init_tau (float): Initial value for the learnable temperature parameter tau.
        init_b (float): Initial value for the learnable bias parameter b.
    """
    def __init__(self, opt, max_tau=10.0, alpha=0.5, init_tau=1.0, init_b=0.0):
        super(DCDLoss, self).__init__()
        self.params = LearnableParams(init_tau=init_tau, init_b=init_b)
        self.max_tau = max_tau
        self.alpha = alpha
        
        self.embed_s = nn.Linear(opt.s_dim, opt.feat_dim)
        self.embed_t = nn.Linear(opt.t_dim, opt.feat_dim)
    
    def forward(self, f_s, f_t):
        """
        Compute the DCD loss between student features (f_s) and teacher features (f_t).
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        f_s = F.normalize(f_s, dim=1)  
        f_t = F.normalize(f_t, dim=1)  
        
        n = f_s.size(0)
        
        tau = self.params.tau.exp().clamp(0, self.max_tau)
        b = self.params.b
        
        logits = torch.mm(f_s, f_t.t()) * tau + b
        labels = torch.arange(n).to(logits.device)
        
        # contrastive loss
        contrastive_loss = F.cross_entropy(logits, labels)
        p1 = F.log_softmax(logits, dim=1)
        p2 = F.softmax(logits, dim=0).t()
        
        # # KL-divergence loss for distribution invariance
        invariance_loss = F.kl_div(p1, p2, reduction="batchmean")
        
        total_loss = contrastive_loss + self.alpha * invariance_loss
        return total_loss


class LearnableParams(nn.Module):
    """
    Hack to make nn.Parameter into an nn.Module
    """
    def __init__(self, init_tau=1.0, init_b=0.0):
        super(LearnableParams, self).__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau))  # Learnable temperature parameter
        self.b = nn.Parameter(torch.tensor(init_b))      # Learnable bias parameter

    def forward(self):
        # This module doesn't need to do anything in forward, since it's just holding parameters.
        pass
