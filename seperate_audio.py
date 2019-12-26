import numpy as np
from glob import glob
import subprocess
import os
import shutil
import json
import audiofile
from concurrent.futures import ProcessPoolExecutor

"""
FFMPEG convert all mp4 to aac
ls *mp4 | parallel --dry-run "ffmpeg -i {} -vn -acodec copy {/.}.aac"
Do above command parrallel for all folders
for f in *; do ls $f/*mp4 | parallel "ffmpeg -i {} -vn -acodec copy $f/{/.}.aac"; done 
"""

def seperate_real_and_fake_audio(f):
    metadata = os.path.join(f, 'metadata.json')
    os.makedirs(os.path.join(f, 'audio', 'real'), exist_ok=True)
    os.makedirs(os.path.join(f, 'audio', 'fake'), exist_ok=True)
    with open(metadata, 'r') as w:
        d = json.load(w)
        for key,value in d.items():
            original_video = value.get('original', None)
            if original_video:
                # get real audio stream
                real_audio_file = os.path.join(f, original_video.split('.')[0] + '.aac')
                rsig, rfs = audiofile.read(real_audio_file)
                # get fake audio stream
                fake_audio_file = os.path.join(f, key.split('.')[0] + '.aac')
                fsig, ffs = audiofile.read(fake_audio_file)
                # compare
                if not np.array_equal(rsig, fsig):
                    if os.path.exists(fake_audio_file):
                        shutil.move(fake_audio_file, os.path.join(f, 'audio', 'fake'))




if __name__ == '__main__':
    fl = glob('../raw/*')
    with ProcessPoolExecutor(max_workers=60) as executor:
        executor.map(seperate_real_and_fake_audio, fl)


