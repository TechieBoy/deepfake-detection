import numpy as np
import h5py
from audio import read_as_melspectrogram
from audio_hp import audio_config
from tqdm import tqdm
import pandas as pd

ndf = pd.read_csv('audiofakes.txt')
fake_audio = ndf.audio_fake.to_list()
real_audio = set(ndf.real.to_list())

num_fakes = len(fake_audio)
num_real = len(real_audio)
total = num_fakes + num_real
tmean = 0.0
tstd = 0.0

with h5py.File("audio.hdf5", "w") as hdfile:
    hdfile.create_dataset("audio", shape=(total, audio_config.img_w, audio_config.img_h), dtype=np.float32)
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
