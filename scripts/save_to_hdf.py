import numpy as np
import h5py
import os
from tqdm import tqdm

fake_folder = "/home/teh_devs/deepfake/dataset/of/fake"
real_folder = "/home/teh_devs/deepfake/dataset/of/real"


num_fakes = 351653
num_real = 69144
total = num_fakes + num_real
print(total)

tmean = np.zeros(2, dtype=np.float64)
tstd = np.zeros(2, dtype=np.float64)

with h5py.File("/home/teh_devs/usb/three_frame_flow.hdf5", "w") as hdfile:
    hdfile.create_dataset("flow", shape=(total, 300, 300, 2), dtype=np.float32)
    hdfile.create_dataset("num_fake", shape=(1,), dtype=np.uint32)
    hdfile.create_dataset("num_real", shape=(1,), dtype=np.uint32)
    hdfile.create_dataset("mean", shape=(2,), dtype=np.float64)
    hdfile.create_dataset("std", shape=(2,), dtype=np.float64)
    hdfile["num_fake"][...] = num_fakes
    hdfile["num_real"][...] = num_real
    count = 0
    for folder in [fake_folder, real_folder]:
        for imf in tqdm(sorted(os.listdir(folder)), ncols=0):
            components = imf.split(".")[0].split("_")
            diff = int(components[-1]) - int(components[-2])
            if diff == 3:
                path = os.path.join(folder, imf)
                with open(path, "rb") as f:
                    magic, = np.fromfile(f, np.float32, count=1)
                    if 202021.25 != magic:
                        print(f"Magic number incorrect. Invalid .flo file {path}, {count}")
                    else:
                        w, h = np.fromfile(f, np.int32, count=2)
                        # print(f"Reading {w} x {h} flo file")
                        data = np.fromfile(f, np.float32, count=2 * w * h)
                        data2D = np.resize(data, (h, w, 2))
                        hdfile["flow"][count, ...] = data2D[None]
                        tmean += data2D.reshape(-1, 2).mean(0)
                        tstd += data2D.reshape(-1, 2).std(0)
                        count += 1

    tmean /= np.full_like(tmean, total)
    tstd /= np.full_like(tstd, total)
    hdfile["mean"][...] = tmean[None]
    hdfile["std"][...] = tstd[None]


# for folder in [fake_folder, real_folder]:
#     count = 0
#     for imf in tqdm(sorted(os.listdir(folder)), ncols=0):
#         components = imf.split(".")[0].split("_")
#         diff = int(components[-1]) - int(components[-2])
#         if diff == 3:
#             count += 1
#     print(count)