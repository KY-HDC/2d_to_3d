import os
import cv2
from tqdm import tqdm

bias_pixel = 50

directory = "/data2/gayrat/selection-task/202207151_pp/188"

new_directory = "7K_raw_images"

if not os.path.exists(new_directory):
    os.mkdir(new_directory)


for file_name in os.listdir(directory):
    sub_dir_path = directory + '/' + file_name
    if (os.path.isdir(sub_dir_path)):
        for image_name in tqdm(os.listdir(sub_dir_path)):
            if image_name[-4:] == '.jpg':
                img = cv2.imread(sub_dir_path + '/' + image_name)
                h, w, ch = img.shape
                
                #croped_nec_img = img[0 : h, w-h : w]
                
                #width = 1024
                #height = 1024
                #points = (width, height)
                
                #img = cv2.resize(croped_nec_img, points, interpolation= cv2.INTER_AREA)
                copied_image_path = new_directory + '/' + image_name
                cv2.imwrite(copied_image_path, img)


