# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import model_dict
from datasets import get_cifar100_dataloaders, get_cifar10_dataloaders

MAX_DIFF = 3.0
METHOD_PATHS = {
    'Vanilla': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\models\wrn_40_1_vanilla\ckpt_epoch_240.pth", 
    'Teacher': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\models\wrn_40_2_vanilla\ckpt_epoch_240.pth",
    'KD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\KD_cifar100_S_wrn_40_1_T_wrn_40_2_r_0.1_a_0.9_b_0.0_trial_1\ckpt_epoch_240.pth",
    'FitNets': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\HINT_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_100.0_trial_1\ckpt_epoch_240.pth",
    'AT': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\ATTENTION_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1000.0_trial_1\ckpt_epoch_240.pth",
    'SP': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\SIMILARITY_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_3000.0_trial_1\ckpt_epoch_240.pth",
    'CC': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\CORRELATION_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_0.02_trial_1\ckpt_epoch_240.pth",
    'VID': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\VID_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'RKD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\RKD_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'PKT': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\PKT_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_30000.0_trial_1\ckpt_epoch_240.pth",
    'ABOUND': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\ABOUND_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'FT': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\FACTOR_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_200.0_trial_1\ckpt_epoch_240.pth",
    'NST': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\NST_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_50.0_trial_1\ckpt_epoch_240.pth",
    'CRD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\CRD_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'CRD+KD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\CRD_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_1.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'RRD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\RRD_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_1.0_trial_1\ckpt_epoch_240.pth",
    'RRD+KD': r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\RRD_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.9_b_1.0_trial_1\ckpt_epoch_240.pth"
}


def load_model(model_path, n_cls, model_name):
    """Load model from checkpoint"""
    print(f'==> Loading model: {model_name}')
    model = model_dict[model_name](num_classes=n_cls)
    try:
        model.load_state_dict(torch.load(model_path)['model'])
    except:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('Model loaded successfully!')
    return model

def load_dataset(dataset):
    """Loads the dataset and returns the validation data loader and number of classes."""
    if dataset == 'cifar100':
        _, val_loader = get_cifar100_dataloaders(batch_size=256, num_workers=8, is_instance=False)
        num_classes = 100
    elif dataset == 'cifar10':
        _, val_loader = get_cifar10_dataloaders(batch_size=256, num_workers=8, is_instance=False)
        num_classes = 10
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
    print("Data loaded successfully!")
    return val_loader, num_classes

def normalize(logit):
    """Normalize logits to zero mean and unit variance."""
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / stdv

def get_output_metric(model, val_loader, num_classes=100, ifstand=False):
    """Extract the mean class-wise logits from the model."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for _, (data, labels) in tqdm(enumerate(val_loader)):
            data = data.cuda()
            labels = labels.cuda()
            outputs = model(data)
            preds = normalize(outputs) if ifstand else outputs
            all_preds.append(preds.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())

    all_preds = np.concatenate(all_preds, 0)
    all_labels = np.concatenate(all_labels, 0)

    matrix = np.zeros((num_classes, num_classes))
    cnt = np.zeros((num_classes, 1))

    for p, l in zip(all_preds, all_labels):
        cnt[l, 0] += 1
        matrix[l] += p

    matrix /= cnt
    return matrix

def get_teacher_student_diff(tea_path, stu_path, tea_name, stu_name, dataset='cifar100', max_diff=MAX_DIFF, ifstand=False, filename='correlation.png'):
    """Compute and visualize the absolute difference between teacher and student logits."""

    # Load dataset
    val_loader, num_classes = load_dataset(dataset)

    # Load teacher and student models
    model_t = load_model(tea_path, num_classes, tea_name)
    model_s = load_model(stu_path, num_classes, stu_name)

    if torch.cuda.is_available():
        model_t.cuda()
        model_s.cuda()

    print("Teacher and student models loaded successfully!")

    # Compute class-wise mean logits
    mt = get_output_metric(model_t, val_loader, num_classes, ifstand=ifstand)
    ms = get_output_metric(model_s, val_loader, num_classes, ifstand=ifstand)

    # Compute absolute difference
    diff = np.abs(ms - mt)
    np.fill_diagonal(diff, 0)

    print(f'Max Difference: {diff.max()}, Index: {diff.argmax()}')
    print(f'Mean Difference: {diff.mean()}')

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(diff, vmin=0, vmax=max_diff, cmap="Blues", cbar_kws={"use_gridspec": False, "location": "left"})
    plt.savefig(filename, dpi=300, format='png', bbox_inches='tight')
    plt.show()

    return diff

if __name__ == "__main__":
    opt = argparse.Namespace()
    opt.dataset = 'cifar100'
    opt.method = 'NST'
    
    opt.model_s = 'wrn_40_1'
    opt.path_s = r'\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\student\student_model\NST_cifar100_S_wrn_40_1_T_wrn_40_2_r_1_a_0.0_b_50.0_trial_1\wrn_40_1_best.pth' #wrn_40_1_best

    opt.model_t = 'wrn_40_2'
    opt.path_t = r"\\rds.imperial.ac.uk\RDS\user\ng1523\home\code\distillers\cifar\save\models\wrn_40_2_vanilla\ckpt_epoch_240.pth"
    
    filename = f"correlation_{opt.method}_{opt.model_s}_{opt.model_t}.png"
    
    get_teacher_student_diff(opt.path_t, 
                             opt.path_s, 
                             opt.model_t, 
                             opt.model_s,
                             dataset=opt.dataset, 
                             max_diff=MAX_DIFF, 
                             ifstand=True,
                             filename=filename,
                             )
