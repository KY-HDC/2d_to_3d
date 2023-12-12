import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt

data_dir='/data2/gayrat/vs-projects/MY_GANs/datasets/CT-0'

def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan

def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(path):
    """Read and resize volume"""
    # Read scan
    volume = read_nifti_file(path)
    # Normalize
    volume = normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


paths = [os.path.join(os.getcwd(), data_dir, x) for x in os.listdir(data_dir)]

# paths = paths[:5]

path_len = len(paths)

abnormal_scans = np.array([process_scan(path) for path in paths])

save_dir = 'vid_dataset_256'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

count_img = 0
for i in tqdm(range(0, path_len)):

    seq_imgs = abnormal_scans[i]
    
    subdir = f'Patient_{i}'
    if not os.path.exists(subdir):
        os.mkdir(subdir)

    for j in range(0, seq_imgs.shape[2]):

        image = np.squeeze(seq_imgs[:, :, j])
        
        image = cv2.resize(image, (256, 256), interpolation= cv2.INTER_AREA)

        # plt.imsave(os.path.join(save_dir, f'{count_img}.jpg'), image, cmap='gray')
        plt.imsave(os.path.join(subdir, f'{count_img}.jpg'), image, cmap='gray')

        count_img+=1

# os.path.join(save_dir, subdir)

print('Done.....')
