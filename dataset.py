import json
from torch.utils.data import Dataset
from PIL import Image


class RubDataset(Dataset):
    def __init__(self, json_path, transform):
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.dataset[idx]['path'])
        img = self.transform(img)
        label = self.dataset[idx]['label']
        return img, label

    def __len__(self):
        return len(self.dataset)
