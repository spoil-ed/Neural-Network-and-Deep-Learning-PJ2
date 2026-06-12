"""CIFAR-10 data loaders.

Standard CIFAR-10 normalization + light augmentation (random crop with 4px
padding, horizontal flip) for the training set; deterministic eval transform
for the test set.
"""
import os
import torch
import torchvision
import torchvision.transforms as T

# Channel statistics computed over the CIFAR-10 training set.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'data')


def _augment_mode(augment):
    if augment is True:
        return 'standard'
    if augment is False:
        return 'none'
    return augment


def get_transforms(train, augment='standard'):
    norm = T.Normalize(CIFAR_MEAN, CIFAR_STD)
    mode = _augment_mode(augment)
    if train and mode == 'standard':
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
        ])
    if train and mode == 'cutout':
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            norm,
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3),
                            value='random'),
        ])
    if train and mode != 'none':
        raise ValueError(f'unknown augment mode {mode}')
    return T.Compose([T.ToTensor(), norm])


def get_strong_transforms():
    """Heavy augmentation: crop+flip + AutoAugment(CIFAR10 policy) + Cutout."""
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.25), value='random'),  # Cutout-style
    ])


def get_strong_loaders(root=_DEFAULT_ROOT, batch_size=128, num_workers=12):
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=get_strong_transforms())
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True,
        transform=get_transforms(train=False))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


def get_loaders(root=_DEFAULT_ROOT, batch_size=128, num_workers=8,
                augment='standard'):
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True,
        transform=get_transforms(train=True, augment=augment))
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True,
        transform=get_transforms(train=False))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
