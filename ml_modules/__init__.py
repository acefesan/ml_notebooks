"""
ML Modules Package
Modular PyTorch implementations for MNIST and CIFAR-10 classification
"""

from .models import MNISTCNN, CIFAR10CNN
from .data import get_mnist_loaders, get_cifar10_loaders


__all__ = [
    'MNISTCNN', 'CIFAR10CNN',
    'get_mnist_loaders', 'get_cifar10_loaders',
    'train_model', 'evaluate_model',
    'show_samples', 'show_predictions', 'plot_training_history', 'compare_models'
]