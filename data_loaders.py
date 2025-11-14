import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Tuple, Optional
import numpy as np


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing MoE models."""

    def __init__(
        self,
        num_samples: int = 1000,
        input_size: int = 784,
        num_classes: int = 10,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes

        # Generate synthetic data with patterns
        self.data = torch.randn(num_samples, input_size)

        # Generate labels based on data patterns
        # Different experts should specialize in different patterns
        patterns = torch.randn(num_classes, input_size)
        similarities = torch.matmul(self.data, patterns.T)
        self.labels = torch.argmax(similarities, dim=1)

        # Add noise
        self.data += torch.randn_like(self.data) * noise_level

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MultiTaskDataset(Dataset):
    """
    Dataset with multiple tasks to encourage expert specialization.

    Each task is a different classification problem on the same input space.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        input_size: int = 784,
        num_tasks: int = 3,
        classes_per_task: int = 10,
        seed: int = 42
    ):
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.num_tasks = num_tasks

        # Shared input
        self.data = torch.randn(num_samples, input_size)

        # Multiple task labels
        self.task_labels = {}
        for task_id in range(num_tasks):
            # Each task has different patterns
            patterns = torch.randn(classes_per_task, input_size)
            similarities = torch.matmul(self.data, patterns.T)
            self.task_labels[task_id] = torch.argmax(similarities, dim=1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Return data and all task labels
        labels = {task_id: labels[idx] for task_id, labels in self.task_labels.items()}
        return self.data[idx], labels


def get_mnist_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_dir: str = './data',
    augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST data loaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Transformations
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    # Split training into train and validation
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_cifar10_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_dir: str = './data',
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Transformations
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    # Split training into train and validation
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_cifar100_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_dir: str = './data',
    augmentation: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Similar to CIFAR-10 but with 100 classes
    if augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761)
            )
        ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        transform=train_transform,
        download=True
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        transform=test_transform,
        download=True
    )

    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_synthetic_loaders(
    batch_size: int = 32,
    num_samples: int = 1000,
    input_size: int = 784,
    num_classes: int = 10,
    train_split: float = 0.8,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get synthetic data loaders.

    Returns:
        train_loader, test_loader
    """
    dataset = SyntheticDataset(
        num_samples=num_samples,
        input_size=input_size,
        num_classes=num_classes,
        seed=seed
    )

    # Split into train and test
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


def get_data_loaders(
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_dir: str = './data',
    augmentation: bool = False,
    **kwargs
):
    """
    Universal data loader getter.

    Args:
        dataset_name: Name of dataset ('mnist', 'cifar10', 'cifar100', 'synthetic')
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Fraction of data for training
        data_dir: Directory to store data
        augmentation: Whether to use data augmentation
        **kwargs: Additional dataset-specific arguments

    Returns:
        train_loader, val_loader (or test_loader for synthetic), test_loader
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'mnist':
        return get_mnist_loaders(batch_size, num_workers, train_split, data_dir, augmentation)
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers, train_split, data_dir, augmentation)
    elif dataset_name == 'cifar100':
        return get_cifar100_loaders(batch_size, num_workers, train_split, data_dir, augmentation)
    elif dataset_name == 'synthetic':
        return get_synthetic_loaders(
            batch_size=batch_size,
            train_split=train_split,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class FlattenTransform:
    """Transform to flatten images for fully-connected MoE models."""

    def __call__(self, x):
        return x.view(-1)


def get_flattened_mnist_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST loaders with flattened images (for FC networks).

    Returns:
        train_loader, val_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784
    ])

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )

    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
