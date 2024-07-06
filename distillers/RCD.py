# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCDLoss(nn.Module):
    """
    Relational Consistency for Knowledge Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of instances in queue
        opt.nce_t_s: the temperature (default: 0.04)
        opt.nce_t_t: the temperature (default: 0.1)
    """
    def __init__(self, opt):
        super(RCDLoss, self).__init__()
        self.nce_k = opt.nce_k
        self.nce_t_s = opt.nce_t_s
        self.nce_t_t = opt.nce_t_t
        
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        
        self.register_buffer("queue", torch.randn(opt.nce_k, opt.feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, f_s, f_t):
        """
        Compute the RCD loss between student features (f_s) and teacher features (f_t).
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        f_s = F.normalize(f_s, dim=1)  
        f_t = F.normalize(f_t, dim=1)  

        l_s = torch.einsum("nc,kc->nk", [f_s, self.queue.clone().detach()])
        l_t = torch.einsum("nc,kc->nk", [f_t, self.queue.clone().detach()])

        loss = - torch.sum(F.softmax(l_t.detach() / self.nce_t_t, dim=1) *
                           F.log_softmax(l_s / self.nce_t_s, dim=1), dim=1).mean()

        self._dequeue_and_enqueue(f_t)

        return loss
    
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue the oldest batch of features and enqueue the current batch's keys.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        effective_batch_size = min(batch_size, self.nce_k - ptr)
        self.queue[ptr: ptr + effective_batch_size] = keys[:effective_batch_size]
        ptr = (ptr + effective_batch_size) % self.nce_k 
        self.queue_ptr[0] = ptr

        
class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x
    
    
class Normalize(nn.Module):
    """Normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
