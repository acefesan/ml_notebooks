# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a machine learning notebooks repository focused on learning ML concepts following Andrej Karpathy's lectures. The primary project is a PyTorch CNN implementation for MNIST digit classification optimized for Apple Silicon (MPS backend).

## Environment Setup
The project uses Poetry for dependency management:
- **Setup**: `poetry install`
- **Activate environment**: `poetry shell`
- **Python version**: 3.9-3.10 (specified in pyproject.toml)

## Key Dependencies
- PyTorch with torchvision for deep learning
- matplotlib, seaborn for visualization
- scikit-learn for metrics and evaluation
- tqdm for progress bars
- numpy <2.0 (compatibility constraint)

## Architecture
The main notebook (`pytorch_mnist_cnn.ipynb`) contains:
- **MNISTCNN class**: Multi-layer CNN with batch normalization and dropout
  - 3 convolutional blocks (32→64→128 channels)
  - Batch normalization and dropout for regularization
  - Fully connected layers with 512→128→10 architecture
- **Apple Silicon optimization**: Uses MPS backend when available
- **Data augmentation**: Random rotation and affine transforms for training

## Device-Specific Considerations
- **MPS backend**: Prioritized for Apple Silicon, with fallback to CUDA/CPU
- **Known limitations**: Adaptive pooling not supported on MPS, uses fixed-size pooling instead
- **Pin memory**: Disabled for MPS (not supported), enabled for other backends

## Model Assets
- `best_mnist_model.pth`: Best performing model weights
- `training_history.pth`: Training metrics and history
- `data/MNIST/`: MNIST dataset files (downloaded automatically)

## Notebook Structure
The notebook follows a standard ML workflow:
1. Environment setup and device detection
2. Data loading with augmentation
3. Model architecture definition
4. Training loop with validation
5. Evaluation with confusion matrix and predictions visualization
6. Model and history persistence