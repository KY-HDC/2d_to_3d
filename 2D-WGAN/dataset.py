import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image



class Custom_Dataset(Dataset):
    def __init__(self, data_dir):
        self.basedir = data_dir
        self.data_dir = os.listdir(data_dir)
        self.transforms = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], 
                                                        std=[0.5])])
    
        self.image_files = [i for i in self.data_dir if i.endswith('.jpg')] 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.basedir +'/'+ self.image_files[index]
    
        image = Image.open(image_path)
        image = self.transforms(image)

        return image
