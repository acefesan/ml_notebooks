"""
Data loading and preprocessing for MNIST and CIFAR-10 datasets
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get MNIST train and test data loaders with appropriate preprocessing.
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/load data
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    # Data preprocessing and augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data preprocessing for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # Determine pin_memory based on device availability
    pin_memory = torch.cuda.is_available()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    # Dataset information
    dataset_info = {
        'name': 'MNIST',
        'num_classes': 10,
        'input_shape': (1, 28, 28),
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'class_names': [str(i) for i in range(10)]
    }
    
    return train_loader, test_loader, dataset_info


def get_cifar10_loaders(batch_size=64, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-10 train and test data loaders with appropriate preprocessing.
    
    Args:
        batch_size (int): Batch size for data loaders (smaller for CIFAR-10)
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/load data
    
    Returns:
        tuple: (train_loader, test_loader, dataset_info)
    """
    # Data preprocessing and augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Data preprocessing for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Dataset information
    dataset_info = {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'input_shape': (3, 32, 32),
        'train_samples': len(train_dataset),
        'test_samples': len(test_dataset),
        'class_names': class_names
    }
    
    return train_loader, test_loader, dataset_info


def get_device():
    """
    Get the best available device for computation.
    Prioritizes MPS (Apple Silicon) > CUDA > CPU
    
    Returns:
        torch.device: The selected device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def print_device_info():
    """Print information about PyTorch and available devices."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    device = get_device()
    if device.type == "mps":
        print("Using Apple Silicon MPS")
    elif device.type == "cuda":
        print("Using CUDA")
    else:
        print("Using CPU")
    
    return device