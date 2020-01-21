import pandas as pd
from shutil import copy
import os
import torch
import csv

df = pd.read_csv("~/deepfake/raw/combined_metadata.csv")

real_videos = df[df.label == "REAL"]
cf = {}
fd = {}


small_folder = "/home/teh_devs/deepfake/small"


for i in range(50):
    folder_name = f"/home/teh_devs/deepfake/raw/dfdc_train_part_{i}"
    two_real = real_videos[real_videos.folder == folder_name].head(2)["index"].tolist()
    for r in two_real:
        # new_folder = os.path.join(small_folder, r)
        # os.mkdir(new_folder)
        # real_video = os.path.join(folder_name, r)
        # copy(real_video, new_folder)
        csf = df[df.original == r]["index"].tolist()
        # for fake in csf:
        #     copy(os.path.join(folder_name, fake), new_folder)
        cf[r] = csf
        fd[r] = folder_name

# torch.save(cf, "real_to_fake_mapping.pickle")
# torch.save(fd, "real_to_folder_mapping.pickle")


with open("rf_mapping.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["real", "fake", "folder"])
    for real, fakes in cf.items():
        for fake in fakes:
            writer.writerow([real, fake, fd[real]])
