import os
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.transform import resize
import torch


class Custom_Dataset(Dataset):
    def __init__(self, data_dir):
        self.ct_paths = [os.path.join(os.getcwd(), data_dir, x) for x in os.listdir(data_dir)]
        self.sp_size = 64
        self.depth = 64
        # self.sp_size = 128
    
    def __read_nifti_file__(self, filepath):
        scan = nib.load(filepath)
        scan = scan.get_fdata()
        return scan
    
    def __normalize__(self, volume):
        min = -1000
        max = 400
        volume[volume < min] = min
        volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
        return volume
    
    def __resize_volume__(self, img):
        desired_depth = 64
        desired_width = 128
        desired_height = 128
        
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
       
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        
        img = ndimage.rotate(img, 90, reshape=False)
        
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        
        img = resize(img, (self.sp_size, self.sp_size, self.depth), mode='constant')

        return img
    
    def __process_scan__(self, path):

        volume = self.__read_nifti_file__(path)
        volume = self.__normalize__(volume)
        volume = self.__resize_volume__(volume)
        return volume

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, index):

        image_path = self.ct_paths[index]
    
        seq_image = np.array(self.__process_scan__(image_path))

        return torch.from_numpy(seq_image).float().view(1, self.sp_size, self.sp_size, self.depth)