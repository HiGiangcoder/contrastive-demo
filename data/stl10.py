import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
import numpy as np

# ======================================================
# Train / Semi-supervised
# ======================================================
def get_stl10_semi(batch_size, seed=42, num_workers=0):
    np.random.seed(seed)

    train_tf = T.Compose([
        T.RandomResizedCrop(96, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    test_tf = T.Compose([
        T.Resize(96),
        T.CenterCrop(96),
        T.ToTensor(),
    ])

    labeled_set = STL10(
        "./data",
        split="train",
        download=True,
        transform=train_tf,
    )

    unlabeled_set = STL10(
        "./data",
        split="unlabeled",
        download=True,
        transform=train_tf,
    )

    test_set = STL10(
        "./data",
        split="test",
        download=True,
        transform=test_tf,
    )

    labeled_loader = DataLoader(
        labeled_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    unlabeled_loader = DataLoader(
        unlabeled_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return labeled_loader, unlabeled_loader, test_loader


# ======================================================
# Evaluation ONLY
# ======================================================
def get_stl10_eval(batch_size, num_workers=0):
    test_tf = T.Compose([
        T.Resize(96),
        T.CenterCrop(96),
        T.ToTensor(),
    ])

    test_set = STL10(
        "./data",
        split="test",
        download=True,
        transform=test_tf,
    )

    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
