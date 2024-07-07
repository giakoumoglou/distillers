# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
import math


eps = 1e-7


class RCDv2Loss(nn.Module):
    def __init__(self, opt):
        super(RCDv2Loss, self).__init__()
        self.embed_s = Embed(opt.s_dim, opt.feat_dim)
        self.embed_t = Embed(opt.t_dim, opt.feat_dim)
        self.contrast = ContrastMemory(opt.feat_dim, opt.n_data, opt.nce_k, opt.nce_t, opt.nce_m)
        self.nce_t_s = opt.nce_t_s
        self.nce_t_t = opt.nce_t_t

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        
        l_s, l_t = self.contrast(f_s, f_t, idx, contrast_idx)

        loss = -torch.sum(F.softmax(l_t.detach() / self.nce_t_t, dim=1) *
                          F.log_softmax(l_s / self.nce_t_s, dim=1), dim=1).mean()

        return loss


class Embed(nn.Module):
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
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class ContrastMemory(nn.Module):
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

    @torch.no_grad()
    def update_memory(self, v, y):
        momentum = self.params[4].item()

        with torch.no_grad():
            l_pos = torch.index_select(self.memory_v1, 0, y.view(-1))
            l_pos.mul_(momentum)
            l_pos.add_(torch.mul(v, 1 - momentum))
            l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_v1 = l_pos.div(l_norm)
            self.memory_v1.index_copy_(0, y, updated_v1)

    def forward(self, v1, v2, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_v1 = self.params[2].item()
        Z_v2 = self.params[3].item()
        momentum = self.params[4].item()
        
        batchSize = v1.size(0)
        outputSize = self.memory_v1.size(0)
        inputSize = self.memory_v1.size(1)

        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))

        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))

        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))

        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()

        return out_v1, out_v2


class AliasMethod(object):
    def __init__(self, probs):
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj


if __name__ == "__main__":
    class Options:
        def __init__(self, s_dim, t_dim, feat_dim, n_data, nce_k, nce_t, nce_m, nce_t_s, nce_t_t):
            self.s_dim = s_dim
            self.t_dim = t_dim
            self.feat_dim = feat_dim
            self.n_data = n_data
            self.nce_k = nce_k
            self.nce_t = nce_t
            self.nce_m = nce_m
            self.nce_t_s = nce_t_s
            self.nce_t_t = nce_t_t

    f_s = [torch.randn(64, 32, 32, 32), torch.randn(64, 64, 32, 32), torch.randn(64, 128, 16, 16), torch.randn(64, 256, 8, 8), torch.randn(64, 256)]
    f_t = [torch.randn(64, 32, 32, 32), torch.randn(64, 64, 64, 64), torch.randn(64, 256, 32, 32), torch.randn(64, 512, 16, 16), torch.randn(64, 512)]

    # Select the last elements of f_s and f_t
    f_s = f_s[-1].cuda()
    f_t = f_t[-1].cuda()

    # Automatically determine channel shapes
    s_dim = f_s.size(1)
    t_dim = f_t.size(1)
    feat_dim = 128  # Example feature dimension for projection space
    n_data = 10000  # Example total number of data points
    nce_k = 16384  # Example number of negatives
    nce_t = 0.07  # Example temperature for contrast memory
    nce_m = 0.5  # Example momentum for contrast memory
    nce_t_s = 0.07  # Example temperature for student
    nce_t_t = 0.07  # Example temperature for teacher

    opt = Options(s_dim=s_dim, t_dim=t_dim, feat_dim=feat_dim, n_data=n_data, nce_k=nce_k, nce_t=nce_t, nce_m=nce_m, nce_t_s=nce_t_s, nce_t_t=nce_t_t)

    # Instantiate CRDLoss
    crd_loss = RCDv2Loss(opt).cuda()

    # Sample indices
    batch_size = f_s.size(0)
    idx = torch.randint(0, opt.n_data, (batch_size,)).cuda()

    # Compute the loss
    loss = crd_loss(f_s, f_t, idx)
    
    print("Computed CRD Loss:", loss.item())


