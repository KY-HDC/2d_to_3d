import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomD(Dataset):

    def __init__(self, data):
        self.base = data
        self.data = os.listdir(data)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.images = [i for i in self.data if i.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.base + self.images[index]
        image = Image.open(image_path)
        image = self.transforms(image)

        return image

training_data = CustomD()

# Generator
# Disriminator
# loss and optimizer
# start training
# save checkpoint
