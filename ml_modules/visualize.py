import torch
from matplotlib import pyplot as plt


# Visualize some sample data

# /Users/adriansanchez/src/ml_notebooks/.venv/lib/python3.9/site-packages/torch/utils/data/dataloader.py:683:
#  UserWarning: 'pin_memory' argument is set as true but not supported on MPS now,
#  then device pinned memory won't be used.
#  warnings.warn(warn_msg)
def show_mnist_samples(train_loader):
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(images[i].squeeze(), cmap='gray')
        axes[row, col].set_title(f'Label: {labels[i]}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

    return images, labels


# Visualize CIFAR-10 samples
def show_cifar_samples(loader_cifar, cifar10_classes):
    dataiter = iter(loader_cifar)
    images, labels = next(dataiter)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        row, col = i // 5, i % 5
        img = images[i]
        # Denormalize for display
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        axes[row, col].imshow(img.permute(1, 2, 0))
        axes[row, col].set_title(f'{cifar10_classes[labels[i]]}')
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()

    return images, labels


# Show CIFAR-10 predictions
def show_cifar_predictions(model, test_loader, device,cifar10_classes, num_samples=12):
    model.eval()
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        predictions = outputs.argmax(dim=1).cpu()
    
    for i in range(num_samples):
        img = images[i]
        # Denormalize for display
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        pred = predictions[i].item()
        actual = labels[i].item()
        color = 'green' if pred == actual else 'red'
        axes[i].set_title(f'Pred: {cifar10_classes[pred]}\nActual: {cifar10_classes[actual]}', color=color, fontsize=8)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()