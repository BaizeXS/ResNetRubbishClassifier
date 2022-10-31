import json

import torch.cuda
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# Global Variables
lr = 1e-4
BATCH_SIZE = 64
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


class RubDataset(Dataset):
    def __init__(self, json_path, transform):
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)

    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx]['path'])
        label = self.dataset[idx]['label']
        return img, label

    def __len__(self):
        return len(self.dataset)
