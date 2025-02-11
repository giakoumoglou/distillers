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
from sklearn.manifold import TSNE
from tqdm import tqdm

from models import model_dict
from datasets import get_cifar100_dataloaders, get_cifar10_dataloaders

COLORMAP = 'tab20c'
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

def load_model(path, num_classes, model_name):
    """Loads a model from a given checkpoint path."""
    print(f'==> Loading model: {model_name}')
    model = model_dict[model_name](num_classes=num_classes)
    
    try:
        state_dict = torch.load(path)
        model.load_state_dict(state_dict.get('model', state_dict))
    except Exception as e:
        print(f"Error loading model: {e}. Attempting CPU loading.")
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict.get('model', state_dict))

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

def extract_features(dataloader, model, device):
    """Extracts features from the dataset using the provided model."""
    model.eval()
    features, labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Extracting Features"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            features.append(outputs.cpu().numpy())
            labels.append(targets.cpu().numpy())

    total_time = time.time() - start_time
    print(f"Feature extraction completed in {total_time:.2f} seconds.")

    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def plot_tsne(features, labels, filename, num_classes):
    """Computes and plots t-SNE visualization of feature embeddings without axes, legends, or text."""
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=COLORMAP, alpha=0.6)
    
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.savefig(f"{filename}.png", dpi=300, format='png', bbox_inches='tight')
    plt.show()

    # Optional: First 10 classes
    if num_classes > 20:
        print("Computing t-SNE for the first 20 classes...")
        selected_indices = labels < 20
        tsne_results_sub = tsne.fit_transform(features[selected_indices])
        labels_sub = labels[selected_indices]

        plt.figure(figsize=(12, 10))
        plt.scatter(tsne_results_sub[:, 0], tsne_results_sub[:, 1], c=labels_sub, cmap=COLORMAP, alpha=0.6)
        
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        plt.savefig(f"{filename}_10classes.png", dpi=300, format='png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    opt = argparse.Namespace()
    opt.method = 'RRD'

    #===========DO NOT CHANGE===========
    opt.dataset = 'cifar100'
    opt.model_s = 'wrn_40_1'
    opt.path_s = METHOD_PATHS[opt.method]
    
    opt.model_t = 'wrn_40_2'
    opt.filename = f"tsne_{opt.method}_{opt.model_s}_{opt.model_t}"

    # Load dataset
    val_loader, num_classes = load_dataset(opt.dataset)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.path_s, num_classes, opt.model_s)
    model.to(device)

    # Extract features
    features, labels = extract_features(val_loader, model, device)

    # t-SNE visualization
    plot_tsne(features, labels, opt.filename, num_classes)
