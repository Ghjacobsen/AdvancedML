"""MNIST data loading for ensemble VAE geometry experiments.

Provides the same subset used in the course handout:
- 3 classes (digits 0, 1, 2)
- 2048 training samples
- Non-binarized (float32, [0, 1])
"""

import torch
import torch.utils.data
from torchvision import datasets, transforms


def subsample(data, targets, num_data: int, num_classes: int):
    """Subsample MNIST to first num_classes classes and num_data points."""
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)


def get_data_loaders(
    batch_size: int = 32,
    num_train_data: int = 2048,
    num_classes: int = 3,
    data_dir: str = "data/",
):
    """Get train and test data loaders for the MNIST subset.

    Returns:
        (train_loader, test_loader, train_data, test_data)
    """
    train_tensors = datasets.MNIST(
        data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )
    test_tensors = datasets.MNIST(
        data_dir, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()])
    )

    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets, num_train_data, num_classes)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_data, test_data
