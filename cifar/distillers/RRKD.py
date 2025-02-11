# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RRKDLoss(nn.Module):
    """
    Rebundancy Reduction for Knowledge Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
    """
    def __init__(self, opt, lamda=0.0051):
        super(RRKDLoss, self).__init__()
        self.lamda = lamda
        
        self.embed_s = nn.Sequential(
            nn.Linear(opt.s_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, opt.feat_dim)
        )

        self.embed_t = nn.Sequential(
            nn.Linear(opt.t_dim, 8192),
            nn.BatchNorm1d(8192),
            nn.ReLU(),
            nn.Linear(8192, opt.feat_dim)
        )
        self.bn = nn.BatchNorm1d(opt.feat_dim, affine=False)
        
    def off_diagonal(self, x):
        """
        Helper function to fetch off-diagonal elements of a matrix.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def forward(self, f_s, f_t):
        """
        Compute loss between student features (f_s) and teacher features (f_t).
        """
        # embed
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        # normalize
        f_s = F.normalize(f_s, dim=1)  
        f_t = F.normalize(f_t, dim=1)  
        
        # cross decorrelation matrix
        c = self.bn(f_s).T @ self.bn(f_t)
        c.div_(f_s.shape[0])
        
        # rebundancy reduction
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() 
        loss = on_diag + self.lamda * off_diag
        return loss