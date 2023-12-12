import os
import cv2
from tqdm import tqdm

path1 = "/data2/gayrat/vs-projects/3d-img-processing/new_pp_dataset"
path2 = "/data2/gayrat/vs-projects/3d-img-processing/CT_64"
dirs = os.listdir( path1 )
print("----------------------------------------------------------")
print("The number images of the current  dataset:", len(dirs))
print("----------------------------------------------------------")
print("               Data-preprocessing started...              ")
print("----------------------------------------------------------")
print("                   Resizing going on                      ")
print("----------------------------------------------------------")
print("")

if not os.path.exists(path2):
    os.makedirs(path2)
# count = 17130
for item in tqdm(dirs):
    if os.path.isfile(path1+'/'+item):
        
    
        image = cv2.imread(path1+'/'+item, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
       
        width = 64
        height = 64
        points = (width, height)
        resized_image = cv2.resize(image, points, interpolation= cv2.INTER_AREA)
        # print(resized_image.shape)
        cv2.imwrite(os.path.join(path2, item[:-4] + '.jpg'), resized_image) # item[:-4], str(count)
        # count +=1
        # break
cv2.waitKey(0)
print("")
print("----------------------------------------------------------")
print("                Data-preprocessing finished               ")
print("----------------------------------------------------------")
print("                       Thank you~                         ")
print("----------------------------------------------------------")
