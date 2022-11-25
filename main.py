import json
import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from dataset import RubDataset

"""
Set Parameters
"""
lr = 1e-4
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Set Transforms for Training and Testing
"""
transform4train = transforms.Compose([
    # Single Crop
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
transform4test = transforms.Compose([
    # Multiple Crop
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

"""
Set Datasets and DataLoader
"""
train_data = RubDataset('', transform4train)
test_data = RubDataset('', transform4test)
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4)