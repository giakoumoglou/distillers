# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class DCDLoss(nn.Module):
    """
    Discriminative and Consistent Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        max_tau (float): Maximum clamp value for the temperature scaling tau.
        alpha (float): Scaling factor for the KL-divergence part of the loss.
        init_tau (float): Initial value for the learnable temperature parameter tau.
        init_b (float): Initial value for the learnable bias parameter b.
        trainable (bool): If False, fix tau and b to opt.nce_t and 0.
    """
    def __init__(self, opt, max_tau=10.0, alpha=0.5, init_tau=1.0, init_b=0.0, trainable=True):
        super(DCDLoss, self).__init__()
        self.max_tau = max_tau
        self.alpha = alpha
        self.trainable = trainable
        
        # Initialize learnable parameters
        self.params = LearnableParams(init_tau=init_tau, init_b=init_b)
        if not trainable:
            self.params.tau = nn.Parameter(torch.tensor(opt.nce_t), requires_grad=False)
            self.params.b = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.embed_s = nn.Linear(opt.s_dim, opt.feat_dim)
        self.embed_t = nn.Linear(opt.t_dim, opt.feat_dim)
    
    def forward(self, f_s, f_t):
        """
        Compute the DCD loss between student features (f_s) and teacher features (f_t).
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        f_s = torch.nn.functional.normalize(f_s, dim=1)  
        f_t = torch.nn.functional.normalize(f_t, dim=1)  
        
        n = f_s.size(0)
        
        tau = self.params.tau.exp().clamp(0, self.max_tau) if self.trainable else self.params.tau
        b = self.params.b if self.trainable else 0.0
        
        logits = torch.mm(f_s, f_t.t()) * tau + b
        labels = torch.arange(n).to(logits.device)
        
        # contrastive loss
        contrastive_loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # KL-divergence loss for distribution invariance
        p1 = torch.nn.functional.log_softmax(logits, dim=1)
        p2 = torch.nn.functional.softmax(logits, dim=0).t()
        invariance_loss = torch.nn.functional.kl_div(p1, p2, reduction="batchmean")
        
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
