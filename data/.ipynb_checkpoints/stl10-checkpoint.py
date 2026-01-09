import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_stl10_semi(batch_size, split="train", seed=42, num_workers=0):
    np.random.seed(seed)

    transform = T.Compose([
        T.RandomResizedCrop(96, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.Resize(96),
        T.CenterCrop(96),
        T.ToTensor(),
    ])

    labeled_set = STL10(
        "./data",
        split="train",
        download=True,
        transform=transform,
    )

    unlabeled_set = STL10(
        "./data",
        split="unlabeled",
        download=True,
        transform=transform,
    )

    test_set = STL10(
        "./data",
        split="test",
        download=True,
        transform=test_transform,
    )

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
