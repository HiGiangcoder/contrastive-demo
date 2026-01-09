import numpy as np
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader, Subset

def get_cifar100(batch_size=256, contrastive=False, num_workers=4):
    if contrastive:
        transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    test_transform = T.ToTensor()

    train_set = CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = CIFAR100(
        "./data",
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, test_loader

def get_cifar100_semi(
    batch_size,
    label_ratio=0.1,
    contrastive=False,
    seed=42,
    num_workers=4
):
    rng = np.random.RandomState(seed)

    if contrastive:
        transform = T.Compose([
            T.RandomResizedCrop(32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ])
    else:
        transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])

    test_transform = T.ToTensor()

    full_train = CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    test_set = CIFAR100(
        "./data",
        train=False,
        download=True,
        transform=test_transform
    )

    num_total = len(full_train)
    num_labeled = int(num_total * label_ratio)

    indices = rng.permutation(num_total)
    labeled_idx = indices[:num_labeled]
    unlabeled_idx = indices[num_labeled:]

    labeled_set = Subset(full_train, labeled_idx)
    unlabeled_set = Subset(full_train, unlabeled_idx)

    labeled_loader = DataLoader(
        labeled_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return labeled_loader, unlabeled_loader, test_loader
