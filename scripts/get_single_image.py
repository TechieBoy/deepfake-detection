import pandas as pd
from glob import glob
from tqdm import tqdm
import os
import pickle

df = pd.read_csv('~/deepfake/raw/combined_metadata.csv')
real = df[df['label'] == 'REAL']['index'].apply(lambda s: s.split('.')[0]).tolist()

which_image = [15,16,14,17,18,20,19,22,21,24,23,26,25,13,11,12,10,28,29,27,30,8,9,7,5,6,4,2,3,1,0]
print(sorted(which_image))

with open('real.txt', 'w') as fe:
    fe.write('Real list\n')
    for r in tqdm(real, ncols=0):
        for i in which_image:
            imag = f'/data/deepfake/real/{r}_face_{i}.jpg'
            if os.path.isfile(imag):
                fe.write(f'{imag}\n')
                break
        else:
            all_images = glob(f'/data/deepfake/real/{r}*.jpg')
            if all_images:
                mid = all_images[len(all_images) // 2]
                fe.write(f'{mid}\n')
