import torch
import torchvision
from torch.utils.data import Dataset
import os
from PIL import Image
torch.manual_seed(3407)
class GetDataset(Dataset):
    def __init__(self, directory, images):
        self.directory = directory
        self.images = images
        self.entries = self._create_entries()


    def _create_entries(self):
        entries = []
        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            #torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((64, 64))
        ])
        names = []
        for filename in self.images:
            f = os.path.join(self.directory, filename)
            image = Image.open(f)
            x = transform(image).to(torch.float)
            # y = int(filename[filename.index('_') + 1])
            y = 1
            entries.append({'x': x, 'y': y})
            names.append(filename)
        return entries

    def __getitem__(self, index):
        entry = self.entries[index]
        return entry['x'], entry['y']

    def __len__(self):
        return len(self.entries)