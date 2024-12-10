import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fastDP import PrivacyEngine  # Ensure fastDP is installed and available
import time
import timm  # Import timm for ViT models

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CIFAR-10 dataset
def get_data_loader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224 for ViT-Large
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)  # Adjust batch size if needed
    return trainloader

# 2. Load ViT-Large model pre-trained on ImageNet
def get_vit_model():
    model = timm.create_model('vit_large_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, 10)  # Modify output layer for 10 CIFAR-10 classes
    return model.to(device)

model = get_vit_model()
for name, param in model.named_parameters():
    print(param.grad)
    print(f"Parameter name: {name}, Shape: {param.shape}")
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
