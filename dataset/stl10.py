from __future__ import print_function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class STL10Instance(datasets.STL10):
    """STL10Instance Dataset for returning images along with their indices."""
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


def get_stl10_dataloaders(batch_size=64, num_workers=4, is_instance=False):
    """
    Dataloader for the STL-10 dataset.
    """
    data_folder = './data'  # Adjust as necessary

    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2712)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2712)),
    ])

    if is_instance:
        train_set = STL10Instance(root=data_folder,
                                  split='train',
                                  download=True,
                                  transform=train_transform)
        n_data = len(train_set)
    else:
        train_set = datasets.STL10(root=data_folder,
                                   split='train',
                                   download=True,
                                   transform=train_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.STL10(root=data_folder,
                              split='test',
                              download=True,
                              transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size/2),
                             shuffle=False,
                             num_workers=int(num_workers/2))

    if is_instance:
        return train_loader, test_loader, n_data
    else:
        return train_loader, test_loader