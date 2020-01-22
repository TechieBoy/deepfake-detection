import numpy as np
import h5py
from audio import read_as_melspectrogram
from audio_hp import audio_config
from tqdm import tqdm
import pandas as pd

ndf = pd.read_csv("audiofakes.txt")
fake_audio = set(ndf.audio_fake.to_list())
real_audio = set(ndf.real.to_list())

bad_files = [
    "/home/teh_devs/deepfake/raw/dfdc_train_part_15/clmgkufday.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/dyxnxclbhg.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/qawhqaraxg.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/qqtoccrfzs.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/strqdkvswm.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/xqnvbrujjo.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_25/ggjareyaqd.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_25/ihqurgivsl.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_25/ykpvzfdqpw.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_26/hycdczdeby.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/rjudlivnao.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/rsvtkzxufe.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/vebpwqhssp.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/xsrbccytqi.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_45/ctpexqamtx.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_5/inkxxqwrzi.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_5/taqnnsyxip.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_15/uyogufqlec.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/afnbrsikom.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/cozgtibuda.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_18/inluuhjteb.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_25/szkoxlypql.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_25/urnqalzaon.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_26/wfdlblolif.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/sxqroedlhr.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/wnqlzimgbg.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_29/zilqngcakp.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_45/rbvqghnbez.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_5/pjibpowymk.mp4",
    "/home/teh_devs/deepfake/raw/dfdc_train_part_5/teehidqtii.mp4",
]
"""
Padding, /home/teh_devs/deepfake/raw/dfdc_train_part_15/gobvnzkjaf.mp4
Padding, /home/teh_devs/deepfake/raw/dfdc_train_part_15/spcowtjevh.mp4
"""
for bf in bad_files:
    if bf in fake_audio:
        fake_audio.remove(bf)
        print("Removing ", bf)
    if bf in real_audio:
        real_audio.remove(bf)
        print("Removing ", bf)

num_fakes = len(fake_audio)
num_real = len(real_audio)
total = num_fakes + num_real
print(total)
tmean = 0.0
tstd = 0.0

with h5py.File("audio.hdf5", "w") as hdfile:
    hdfile.create_dataset(
        "audio", shape=(total, audio_config.img_w, audio_config.img_h), dtype=np.float32
    )
    hdfile.create_dataset("num_fake", shape=(1,), dtype=np.uint32)
    hdfile.create_dataset("num_real", shape=(1,), dtype=np.uint32)
    hdfile.create_dataset("mean", shape=(1,), dtype=np.float64)
    hdfile.create_dataset("std", shape=(1,), dtype=np.float64)
    hdfile["num_fake"][...] = num_fakes
    hdfile["num_real"][...] = num_real
    count = 0
    for audio_list in [fake_audio, real_audio]:
        for audio in tqdm(sorted(audio_list), ncols=0):
            spec = read_as_melspectrogram(audio)
            hdfile["audio"][count, ...] = spec[None]
            tmean += spec.mean()
            tstd += spec.std()
            count += 1

    tmean /= total
    tstd /= total
    hdfile["mean"][...] = tmean
    hdfile["std"][...] = tstd
