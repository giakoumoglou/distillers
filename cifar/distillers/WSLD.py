from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class WSLDLoss(nn.Module):
    """Rethinking soft labels for knowledge distillation: a bias-variance tradeoff perspective (ICLR 2021),
    code from author: https://github.com/bellymonster/Weighted-Soft-Label-Distillation/"""
    def __init__(self, opt):
        super(WSLDLoss, self).__init__()

        self.T = opt.nce_t # (default: 2)
        self.alpha = 2.5
        self.label_smooth = True
        if opt.dataset == 'cifar100':
            self.n_cls = 100 
        elif opt.dataset == 'cifar10':
            self.n_cls = 10
        else:
            assert False

        self.softmax = nn.Softmax(dim=1).cuda()
        self.logsoftmax = nn.LogSoftmax().cuda()

        if self.label_smooth:
            self.hard_loss = cross_entropy_with_label_smoothing
        else:
            self.hard_loss = nn.CrossEntropyLoss().cuda()


    def forward(self, fc_t, fc_s, label):
        s_input_for_softmax = fc_s / self.T
        t_input_for_softmax = fc_t / self.T

        t_soft_label = self.softmax(t_input_for_softmax)

        softmax_loss = - torch.sum(t_soft_label * self.logsoftmax(s_input_for_softmax), 1, keepdim=True)

        fc_s_auto = fc_s.detach()
        fc_t_auto = fc_t.detach()
        log_softmax_s = self.logsoftmax(fc_s_auto)
        log_softmax_t = self.logsoftmax(fc_t_auto)
        one_hot_label = F.one_hot(label, num_classes=self.n_cls).float()
        softmax_loss_s = - torch.sum(one_hot_label * log_softmax_s, 1, keepdim=True)
        softmax_loss_t = - torch.sum(one_hot_label * log_softmax_t, 1, keepdim=True)

        focal_weight = softmax_loss_s / (softmax_loss_t + 1e-7)
        ratio_lower = torch.zeros(1).cuda()
        focal_weight = torch.max(focal_weight, ratio_lower)
        focal_weight = 1 - torch.exp(- focal_weight)
        softmax_loss = focal_weight * softmax_loss

        soft_loss = (self.T ** 2) * torch.mean(softmax_loss)

        hard_loss = self.hard_loss(fc_s, label)

        loss = hard_loss + self.alpha * soft_loss

        return loss


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.0):
    """
    Label smoothing implementation.
    This function is taken from https://github.com/MIT-HAN-LAB/ProxylessNAS/blob/master/proxyless_nas/utils.py
    """
    logsoftmax = nn.LogSoftmax().cuda()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))
