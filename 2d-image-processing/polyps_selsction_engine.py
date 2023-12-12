import os
import pandas as pd
from tqdm import tqdm
import cv2


num = 187
xlsx_file = f'202206271/{num}/20220621_135411_pp.xlsx'

folder = os.listdir(f'202206271/{num}')

file = pd.ExcelFile(xlsx_file)

df = file.parse('Sheet1')
df = pd.DataFrame(df)

patient_ids = df['Patient ID'].tolist()

new_list = []

for i in tqdm(patient_ids):

    for j in folder:

        if  f"'{j[:8]}" == i:
            new_list.append(f'202206271/{num}/{j}/JPG/')

print(len(new_list))

for i in tqdm(new_list):

    final_path = f"202206271_pp/{num}/{i[14:-5]}"
    
    if not os.path.exists(final_path):
        os.makedirs(final_path)

    for j in os.listdir(i):
        
        image = cv2.imread(i+j)

        cv2.imwrite(os.path.join(final_path , j[:-4]+'.jpg'), image)
        
    