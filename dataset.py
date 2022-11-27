import json

from PIL import Image
from torch.utils.data import Dataset


class RubDataset(Dataset):
    def __init__(self, file_path, json_path, transform):
        self.path = file_path
        with open(json_path, 'r') as f:
            self.dataset = json.load(f)
        self.transform = transform

    def __getitem__(self, idx):
        item_path = self.path + "/" + self.dataset[idx]['path']
        image = Image.open(item_path)
        image = self.transform(image)
        label = self.dataset[idx]['label']
        label = int(label)
        return image, label

    def __len__(self):
        return len(self.dataset)
