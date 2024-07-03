from __future__ import print_function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TinyImageNetInstance(datasets.ImageFolder):
    """TinyImageNet Dataset with instance indices."""
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
