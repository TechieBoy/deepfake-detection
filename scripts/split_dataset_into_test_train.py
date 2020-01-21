import pandas as pd
from shutil import move
import os
from tqdm import tqdm


df = pd.read_csv("../../raw/combined_metadata.csv")
real_series = df[df["label"] == "REAL"]["index"].apply(lambda s: s.split(".")[0])

fake_series = df[df["label"] == "FAKE"]["index"].apply(lambda s: s.split(".")[0])
real_folder = "/home/teh_devs/deepfake/dataset/of/real"
fake_folder = "/home/teh_devs/deepfake/dataset/of/fake"

series = [real_series, fake_series]
folders = [real_folder, fake_folder]

base_dir = '/home/teh_devs/deepfake/of/'

for folder in folders:
    if not os.path.isdir(folder):
        os.mkdir(folder)

for s, f in zip(series, folders):
    for name in tqdm(s, ncols=0):
        curr_folder = os.path.join(base_dir, name)
        if not os.path.isdir(curr_folder):
            continue

        for some_file in os.listdir(curr_folder):
            if some_file.split('.')[-1] == 'flo':
                source_loc = os.path.join(curr_folder, some_file)
                move(source_loc, f)
        # for i in range(31):
        #     file_name = f"/home/teh_devs/deepfake/dataset/new/{name}_face_{i}.jpg"
        #     if os.path.isfile(file_name):
        #         move(file_name, f)
        #     else:
        #         print(f"{file_name} does not exist")

# Second pass for missed files (longer videos with more frames)
# for name in real_series:
#     fl = glob(f"/home/teh_devs/deepfake/dataset/new/{name}*")
#     for i in fl:
#         move(i, f)
#         print(f"Moved {i} to {f}")


# print("All remaining are fake")
