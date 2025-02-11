# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class CCDLoss(nn.Module):
    """
    Contrastive Distillation Loss: InfoNCE loss
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of instances in queue
        opt.nce_t: the temperature (default: 0.07)
    """
    def __init__(self, opt):
        super(CCDLoss, self).__init__()
        self.nce_k = opt.nce_k
        self.nce_t = opt.nce_t
        
        self.embed_s = nn.Linear(opt.s_dim, opt.feat_dim)
        self.embed_t = nn.Linear(opt.t_dim, opt.feat_dim)
        
        self.register_buffer("queue", torch.randn(opt.nce_k, opt.feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, f_s, f_t):
        """
        Compute the InfoNCE loss between student features (f_s) and teacher features (f_t).
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)  
        
        f_s = F.normalize(f_s, dim=1)  
        f_t = F.normalize(f_t, dim=1)  
        
        self._dequeue_and_enqueue(f_t)

        l_pos = torch.einsum("nc,nc->n", [f_s, f_t]).unsqueeze(-1)
        l_neg = torch.einsum("nc,kc->nk", [f_s, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_t
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        return loss
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """
        Dequeue the oldest batch of features and enqueue the current batch's keys.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.nce_k % batch_size == 0 # for simplicity
        self.queue[ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.nce_k  # move pointer
        self.queue_ptr[0] = ptr