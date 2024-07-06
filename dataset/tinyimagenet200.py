from __future__ import print_function
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

    
class TinyImageNetInstance(datasets.ImageFolder):
    """
    TinyImageNet Dataset with instance indices.
    """
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_tiny_imagenet_dataloaders(batch_size=128, num_workers=8, is_instance=False):
    """
    Dataloader for the Tiny ImageNet dataset.
    """
    data_folder = './data/tiny-imagenet-200/'
    train_folder = data_folder + '/train'
    val_folder = data_folder + '/val'

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    ])

    if is_instance:
        train_set = TinyImageNetInstance(root=train_folder,
                                         transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.ImageFolder(root=train_folder,
                                         transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.ImageFolder(root=val_folder,
                                    transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader


class TinyImageNetDataset(Dataset):
    """
    TinyImageNet Dataset. not used since we change the structure to match ImageFolder.
    """
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        if self.train:
            self.data = []
            self.labels = []
            classes = os.listdir(os.path.join(root_dir, 'train'))
            for label, cls in enumerate(classes):
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)
        else:
            self.data = []
            self.labels = []
            val_dir = os.path.join(root_dir, 'val', 'images')
            val_annotations = pd.read_csv(os.path.join(root_dir, 'val', 'val_annotations.txt'), 
                                          sep='\t', header=None, 
                                          names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])
            class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train'))))}
            for _, row in val_annotations.iterrows():
                self.data.append(os.path.join(val_dir, row['file_name']))
                self.labels.append(class_to_idx[row['class']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label