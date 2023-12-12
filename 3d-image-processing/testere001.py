import os
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib.pyplot as plt


class Custom_Dataset():
    def __init__(self, data_dir):
        self.ct_paths = [os.path.join(os.getcwd(), data_dir, x) for x in os.listdir(data_dir)]
    
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

        return seq_image

# ----------------------------------------------------------------------------------------------------------------
dataset = Custom_Dataset(data_dir='/data2/gayrat/vs-projects/MY_GANs/3D_image_classification/CT-0')

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

data_dir='/data2/gayrat/vs-projects/MY_GANs/3D_image_classification/CT-0'

paths = [os.path.join(os.getcwd(), data_dir, x) for x in os.listdir(data_dir)]

abnormal_scans = np.array([process_scan(path) for path in paths[:3]])

# print(abnormal_scans[:10].shape)

image = abnormal_scans[0]
plt.imshow(np.squeeze(image[:, :, 30]), cmap="gray")
plt.show()

def plot_slices(num_rows, num_columns, width, height, data):
    """Plot a montage of 20 CT slices"""
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            axarr[i, j].imshow(data[i][j], cmap="gray")
            axarr[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()

# plot_slices(8, 8, 128, 128, abnormal_scans[2][:, :, :64])

