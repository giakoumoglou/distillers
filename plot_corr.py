# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--eval', action='store_true', help='evaluate student and teacher')
    parser.add_argument('--filename', type=str, default='fig', help='name of figure')
    parser.add_argument('--plot_multiple', action='store_true', help='plot multiple images')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_s', type=str, default=None, help='student model snapshot')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    opt = parser.parse_args()
    
    opt.model_s = 'MobileNetV2'
    #opt.path_s = './save/models/wrn_40_1_vanilla/ckpt_epoch_240.pth'
    opt.path_s = './save/student_models/ICD_S_MobileNetV2_T_ResNet50_cifar100_r_1_a_1.0_b_1.0_trial_1/ckpt_epoch_240.pth'
    
    opt.path_t = './save/models/ResNet50_vanilla/ckpt_epoch_240.pth'
    #opt.plot_multiple = True
    
    return opt


def get_teacher_name(model_path):
    """Parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> Loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    try:
        model.load_state_dict(torch.load(model_path)['model'])
    except:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('Teacher model loaded')
    return model

def load_student(model_path, n_cls, model_s):
    print('==> Loading student model')
    model = model_dict[model_s](num_classes=n_cls)
    try:
        model.load_state_dict(torch.load(model_path)['model'])
    except:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('Student model loaded')
    return model


def main():

    opt = parse_option()

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=256, num_workers=8, is_instance=False)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = load_student(opt.path_s, n_cls, opt.model_s)
    
    if torch.cuda.is_available():
        model_t.cuda()
        model_s.cuda()
    
    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    
    # validate teacher and student accuracy
    if opt.eval:
        teacher_acc, _, _ = validate(val_loader, model_t, nn.CrossEntropyLoss(), opt)
        print(f'Teacher accuracy: {teacher_acc.item()}%')
        
        student_acc, _, _ = validate(val_loader, model_s, nn.CrossEntropyLoss(), opt)
        print(f'Student accuracy: {student_acc.item()}%')
    
    # correlation difference norm
    if opt.plot_multiple:
        for i, (data, targets) in enumerate(val_loader):
            if i >= 30:
                break
            with torch.no_grad():
                data = data.cuda()
                targets = targets.cuda()
                
                logits_t = model_t(data, is_feat=True)[1]
                logits_s = model_s(data, is_feat=True)[1]

                corr_t = correlation_matrix_norm(logits_t)
                corr_s = correlation_matrix_norm(logits_s)

                plot_difference(corr_t.detach().cpu().numpy(), corr_s.detach().cpu().numpy(), f'{opt.filename}_{i+1}')
    else:
        for data, targets in val_loader:
            with torch.no_grad():
                data = data.cuda()
                targets = targets.cuda()
                
                logits_t = model_t(data, is_feat=True)[1]
                logits_s = model_s(data, is_feat=True)[1]

                corr_t = correlation_matrix_norm(logits_t)
                corr_s = correlation_matrix_norm(logits_s)
                
                print(f'Difference: Max: {(corr_t.detach().cpu().numpy() - corr_s.detach().cpu().numpy()).max()}, Min: {(corr_t.detach().cpu().numpy() - corr_s.detach().cpu().numpy()).min()}')

                plot_difference(corr_t.detach().cpu().numpy(), corr_s.detach().cpu().numpy(), opt.filename)
                break


def correlation_matrix_norm(logits):
    """Compute the correlation matrix from logits, ensuring each feature has unit variance."""
    logits_centered = logits - logits.mean(dim=0)
    logits_normalized = logits_centered / logits_centered.std(dim=0, keepdim=True)
    correlation = torch.mm(logits_normalized.T, logits_normalized) / logits.size(0)
    return correlation


def correlation_matrix(logits):
    """Compute the correlation matrix from logits, ensuring each feature has unit variance."""
    correlation = torch.mm(logits.T, logits) / logits.size(0)
    return correlation


def plot_difference(corr_t, corr_s, filename='fig'):
    """Plot the difference in correlation matrices as a heatmap."""
    diff_corr = corr_t - corr_s
    plt.figure(figsize=(10, 8))
    im = plt.imshow(diff_corr, cmap='PuOr',)# vmin=-0.3, vmax=0.3)
    plt.colorbar(im)
    plt.xticks(np.arange(0, 100, 4))  
    plt.yticks(np.arange(0, 100, 4))  
    plt.grid(False) 
    plt.savefig(f'{filename}.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()  
    plt.close()


def validate(val_loader, model, criterion, opt):
    """Validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
if __name__ == '__main__':
    main()
