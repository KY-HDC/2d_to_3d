import os
import pydicom
import numpy as np
from PIL import Image, ImageOps


def get_names(path):
    names = []
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in ['.dcm']:
                names.append(filename)

    return names


def convert_dcm_jpg(name):
    im = pydicom.dcmread('ct_dicom_imgs/' + name)  # change folder_name. Rename 'datadicom/'

    im = im.pixel_array.astype(float)

    rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixels
    final_image = np.uint8(rescaled_image)  # integers pixels

    final_image = Image.fromarray(final_image)

    return final_image


names = get_names('ct_dicom_imgs') # change folder_name. Rename 'datadicom/'
for name in names:
    image = convert_dcm_jpg(name)
    image = ImageOps.grayscale(image)
    
    if not os.path.exists('d2j_result_file1'):
        os.makedirs('d2j_result_file1')
    image.save(f'd2j_result_file1/{name[:-4]}.jpg')

