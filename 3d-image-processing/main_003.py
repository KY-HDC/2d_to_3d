import cv2
import os
import re
 
from os.path import isfile, join

pathIn= '/data2/gayrat/vs-projects/3d-img-processing/SCANS'
pathOut = 'video_3d.avi'
fps = 4.0

frame_array = []

files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

# files = files[:30]

files.sort(key=lambda f: int(re.sub('\D', '', f)))

size = (0,0)

for i in range(len(files)):
    filename = pathIn + '/' + files[i]

    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = size[0] + width,size[1] + height


    frame_array.append(img)

print(len(frame_array))

size = int(size[0]/len(files)), int(size[1]/len(files))

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

for i in range(len(frame_array)):
        # print("Wrote file : ", i)
	out.write(frame_array[i])
out.release()
