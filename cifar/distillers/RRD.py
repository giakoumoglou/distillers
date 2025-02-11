# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import math


class RRD_FifoMemory_Loss(nn.Module):
    """
    Relational Representation Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of instances in queue
        opt.nce_t_s: the temperature (default: 0.02)
        opt.nce_t_t: the temperature (default: 0.1)
    """
    def __init__(self, opt):
        super(RRD_FifoMemory_Loss, self).__init__()
        self.nce_k = opt.nce_k
        self.nce_t_s = opt.nce_t_s
        self.nce_t_t = opt.nce_t_t
        
        self.embed_s = nn.Linear(opt.s_dim, opt.feat_dim)
        self.embed_t = nn.Linear(opt.t_dim, opt.feat_dim)
        
        self.register_buffer("queue", torch.randn(opt.nce_k, opt.feat_dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    def forward(self, f_s, f_t):
        """
        Compute the RRD loss between student features (f_s) and teacher features (f_t).
        """
        # forward
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)  
        
        # normalize
        f_s = nn.functional.normalize(f_s, dim=1)  
        f_t = nn.functional.normalize(f_t, dim=1)  

        # enqueue teacher before similarities
        self._dequeue_and_enqueue(f_t)

        # similarities
        out_s = torch.einsum("nc,kc->nk", [f_s, self.queue.clone().detach()])
        out_t = torch.einsum("nc,kc->nk", [f_t, self.queue.clone().detach()])

        # relational loss
        loss = - torch.sum(nn.functional.softmax(out_t.detach() / self.nce_t_t, dim=1) *
                           nn.functional.log_softmax(out_s / self.nce_t_s, dim=1), dim=1).mean()
        
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
        ptr = (ptr + batch_size) % self.nce_k # move pointer
        self.queue_ptr[0] = ptr


class RRD_MomentumMemory_Loss(nn.Module):
    """
    Relational Representation Distillation
    
    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t_s: student temperature
        opt.nce_t_t: teacher temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefore the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self, opt):
        super(RRD_MomentumMemory_Loss, self).__init__()
        self.embed_s = nn.Linear(opt.s_dim, opt.feat_dim)
        self.embed_t = nn.Linear(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.nce_t_t = opt.nce_t_t
        self.nce_t_s = opt.nce_t_s

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The relational loss
        """
        # forward
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)

        # normalize
        f_s = nn.functional.normalize(f_s, dim=1)  
        f_t = nn.functional.normalize(f_t, dim=1)  

        # similarities
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)

        # relational loss
        loss = - torch.sum(nn.functional.softmax(out_t.detach() / self.nce_t_t, dim=1) *
                           nn.functional.log_softmax(out_s / self.nce_t_s, dim=1), dim=1).mean()
        return loss


class RRDLoss(nn.Module):
    """
    Unified RRD Loss that can use either FIFO or Momentum memory
    
    Args:
        opt: Options containing memory_type and other necessary parameters
    """
    def __init__(self, opt):
        super(RRDLoss, self).__init__()
        self.memory_type = getattr(opt, 'memory_type', 'fifo')
        
        # Initialize the appropriate memory mechanism
        if self.memory_type == 'momentum':
            self.memory = RRD_MomentumMemory_Loss(opt)
        else:  # fifo
            self.memory = RRD_FifoMemory_Loss(opt)
    
    def forward(self, f_s, f_t, idx=None, contrast_idx=None):
        """
        Forward pass that delegates to the appropriate memory mechanism
        """
        if self.memory_type == 'momentum':
            return self.memory(f_s, f_t, idx, contrast_idx)
        return self.memory(f_s, f_t)

    @property
    def embed_s(self):
        return self.memory.embed_s

    @property
    def embed_t(self):
        return self.memory.embed_t
    
    
class ContrastMemory(nn.Module):
    """
    Memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory_v1', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_v2', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()

        momentum = self.params[4].item()
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        # update memory
        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v1, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

            ab_pos = torch.index_select(self.memory_v2, 0, y.view(-1))
            ab_pos.mul_(momentum)
            ab_pos.add_(torch.mul(v2, 1 - momentum))
            ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v2 = ab_pos.div(ab_norm)
            self.memory_v2.index_copy_(0, y, updated_v2)

        return out_v1, out_v2


class AliasMethod(object):
    """
    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    """
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj