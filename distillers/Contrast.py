# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastLoss(nn.Module):
    """
    Contrastive Disitllation using InfoNCE
    """
    def __init__(self, opt):
        super(ContrastLoss, self).__init__()
        self.nce_k = opt.nce_k
        self.nce_t = opt.nce_t
        
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        
        self.register_buffer("queue", torch.randn(opt.nce_k, opt.feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_labels", torch.randint(0, 100, (opt.nce_k,)))  # Assuming 10 classes
    
    def forward(self, f_s, f_t, labels, use_valid_negatives=True):
        """
        Compute the contrastive loss between student features (f_s) and teacher features (f_t), filtering negatives by label.
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        f_s = F.normalize(f_s, dim=1)  
        f_t = F.normalize(f_t, dim=1)  

        valid_negatives = (self.queue_labels.unsqueeze(0) != labels.unsqueeze(1)).float()

        l_pos = torch.einsum("nc,nc->n", [f_s, f_t]).unsqueeze(-1)
        
        if use_valid_negatives == True:
            l_neg = torch.einsum("nc,kc->nk", [f_s, self.queue.clone().detach()]) * valid_negatives
        else:
            l_neg = torch.einsum("nc,kc->nk", [f_s, self.queue.clone().detach()])

        logits = torch.cat([l_pos, l_neg], dim=1) / self.nce_t
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(f_t, labels)

        return loss
    
    def _dequeue_and_enqueue(self, keys, keys_labels):
        """
        Dequeue the oldest batch of features and labels and enqueue the current batch's keys and labels.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        effective_batch_size = min(batch_size, self.nce_k - ptr)
        self.queue[ptr: ptr + effective_batch_size] = keys[:effective_batch_size]
        self.queue_labels[ptr: ptr + effective_batch_size] = keys_labels[:effective_batch_size]
        ptr = (ptr + effective_batch_size) % self.nce_k  
        self.queue_ptr[0] = ptr
        
        
class Embed(nn.Module):
    """Embedding module with a single linear layer and normalization."""
    def __init__(self, dim_in, dim_out):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return self.norm(x)


class Normalize(nn.Module):
    """Normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        return x.div(norm)
    
