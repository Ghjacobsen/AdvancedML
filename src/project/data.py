"""
Data loading utilities for binarized MNIST.

This module provides:
- get_mnist_loaders(): Returns train/test DataLoaders for binarized MNIST
- Binarization: pixels > 0.5 -> 1, pixels <= 0.5 -> 0
"""

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def binarize(x: torch.Tensor) -> torch.Tensor:
    """Binarize image tensor: pixels > 0.5 become 1, else 0."""
    return (x > 0.5).float()


def get_mnist_loaders(
    batch_size: int = 128,
    data_dir: str = "./data",
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test DataLoaders for binarized MNIST.
    
    Parameters:
    batch_size: [int] Batch size for training
    data_dir: [str] Directory to download/load MNIST data
    num_workers: [int] Number of workers for data loading (0 for HPC compatibility)
    pin_memory: [bool] Pin memory for faster GPU transfer
    
    Returns:
    train_loader: [DataLoader] Training data loader
    test_loader: [DataLoader] Test data loader
    """
    # Transform: ToTensor (0-1 range), squeeze channel dim, then binarize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze()),  # Remove channel dimension: (1, 28, 28) -> (28, 28)
        transforms.Lambda(binarize),               # Binarize: >0.5 -> 1, else 0
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent batch size
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, test_loader


def get_full_test_loader(
    batch_size: int = 256,
    data_dir: str = "./data",
) -> DataLoader:
    """
    Get test DataLoader with all test samples for evaluation.
    
    Parameters:
    batch_size: [int] Batch size for evaluation
    data_dir: [str] Directory to load MNIST data
    
    Returns:
    test_loader: [DataLoader] Test data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze()),
        transforms.Lambda(binarize),
    ])
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


if __name__ == "__main__":
    # Quick test
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check a batch
    x, y = next(iter(train_loader))
    print(f"Batch shape: {x.shape}")
    print(f"Unique values: {torch.unique(x)}")  # Should be [0, 1]
