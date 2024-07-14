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

from models import model_dict
from datasets import get_cifar100_dataloaders, get_cifar10_dataloaders


def parse_option():
    parser = argparse.ArgumentParser('PyTorch Knowledge Distillation - t-SNE visualization')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10'], help='dataset')
    parser.add_argument('--model', type=str, default='resnet8', choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2', 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50', 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path', type=str, default=None, help='model snapshot')
    parser.add_argument('--filename', type=str, default='fig', help='name of figure')
    opt = parser.parse_args()
    return opt


def load_model(path, n_cls, model):
    print('==> Loading student model')
    model = model_dict[model](num_classes=n_cls)
    try:
        model.load_state_dict(torch.load(path)['model'])
    except:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model'])
    print('Student model loaded')
    return model


def main():

    opt = parse_option()

    # Dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=256, num_workers=8, is_instance=False)
        n_cls = 100
    elif opt.dataset == 'cifar10':
        train_loader, val_loader = get_cifar10_dataloaders(batch_size=256, num_workers=8, is_instance=False)
        n_cls = 10
    else:
        raise NotImplementedError(opt.dataset)

    # Model
    model = load_model(opt.path, n_cls, opt.model)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Extract features
    print("Extracting features from validation set...")
    features, labels = extract_features(val_loader, model, device)

    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(features)

    # Visualization for all classes
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.savefig(f"{opt.filename}.png", dpi=300, format='png', bbox_inches='tight')
    plt.show()
    
    # Conditional visualization for the first 10 classes if more than 10 classes
    if n_cls > 10:
        print("Computing t-SNE for the first 10 classes...")
        indices = labels < 10  # Filter indices for the first 10 classes
        tsne_results_sub = tsne.fit_transform(features[indices])
        labels_sub = labels[indices]

        plt.figure(figsize=(12, 10))
        plt.scatter(tsne_results_sub[:, 0], tsne_results_sub[:, 1], c=labels_sub, cmap='tab10', alpha=0.6)
        plt.savefig(f"{opt.filename}_top10.png", dpi=300, format='png', bbox_inches='tight')
        plt.show()


def extract_features(dataloader, model, device):
    """
    Extracts features from the dataset using the provided model.
    """
    model.eval()
    features = []
    labels = []

    start_time = time.time() 
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            batch_start = time.time()  

            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)
            features.append(output.cpu().numpy())
            labels.append(targets.cpu().numpy())

            batch_end = time.time() 

            if idx % 10 == 0:
                print(f"Processed batch {idx} in {batch_end - batch_start:.2f} seconds.")

    total_time = time.time() - start_time
    print(f"Total feature extraction time: {total_time:.2f} seconds for {len(dataloader)} batches.")

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return features, labels


if __name__ == '__main__':
    main()
