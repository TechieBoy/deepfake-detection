import pandas as pd
from shutil import move
import os
from tqdm import tqdm
from glob import glob
df = pd.read_csv("combined_metadata.csv")
real_series = df[df['label'] == 'REAL']['index'].apply(lambda s: s.split('.')[0])

fake_series = df[df['label'] == 'FAKE']['index'].apply(lambda s: s.split('.')[0])
real_folder = '/raid/deepfake/new/real_new'
fake_folder = '/raid/deepfake/new/fake_new'

series = [real_series, fake_series]
folders = [real_folder, fake_folder]

for s,f in zip(series,folders):
    for name in s:
        for i in range(30):
            file_name = f'/raid/deepfake/new/{name}_face_{i}.jpg'
            if os.path.isfile(file_name):
                move(file_name, f)
            else:
                print(f'{file_name} does not exist')

for s,f in zip(series, folders):
    for name in s:
        fl = glob(f'/raid/deepfake/new/{name}*')
        for i in fl:
            move(i, f)
            print(f"Moved {i} to {f}")
