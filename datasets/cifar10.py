# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


data_folder = './data/'


class CIFAR10Instance(datasets.CIFAR10):
    """
    CIFAR-100 instance dataset
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
    
    
def get_cifar10_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    Get CIFAR-10 data loaders
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    if is_instance:
        train_set = CIFAR10Instance(root=data_folder, download=True, train=True, transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.CIFAR10(root=data_folder, download=True, train=True, transform=train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    test_set = datasets.CIFAR10(root=data_folder, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader
    

class CIFAR10InstanceSample(datasets.CIFAR10):
    """
    CIFAR10Instance+Sample Dataset
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, k=4096, mode='exact', is_sample=True, percent=1.0):
        super().__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 10

        num_samples = len(self.data)
        label = self.targets

        self.cls_positive = [[] for i in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for i in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

        self.cls_positive = np.asarray(self.cls_positive)
        self.cls_negative = np.asarray(self.cls_negative)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
    
        img = Image.fromarray(img)
    
        if self.transform is not None:
            img = self.transform(img)
    
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        if not self.is_sample:
            return img, target, index
        else:
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)[0]
            else:
                raise NotImplementedError(self.mode)
    
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar10_dataloaders_sample(batch_size=128, num_workers=1, k=4096, mode='exact', is_sample=True, percent=1.0):
    """
    Get CIFAR-10 sample data loaders
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR10InstanceSample(root=data_folder, download=True, train=True, transform=train_transform, k=k, mode=mode, is_sample=is_sample, percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder, download=True, train=False, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=int(batch_size/2), shuffle=False, num_workers=int(num_workers/2))

    return train_loader, test_loader, n_data