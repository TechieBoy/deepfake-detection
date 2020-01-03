import os
import subprocess
import json
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from glob import glob
# 
rootdir = '/home/teh_devs/deepfake/raw'
# 
# filename = '/home/teh_devs/deepfake/raw/dfdc_train_part_0/audio/fake/aeqpxjlbwu.aac'
# 
def get_mediainfo_metadata(pathname, filename):
    filename = pathname + "/" + filename
    mediainfo_metadata = subprocess.check_output(['mediainfo', filename])
    mediainfo_metadata = mediainfo_metadata.decode('utf8')
# 
    mediainfo_metadata = mediainfo_metadata.split('\n')
# 
    metadata_row = []
    # 
    for metadata in mediainfo_metadata:
        metainfo = metadata.split(':')
        if len(metainfo) == 2:
            metadata_row.append(metainfo[1])
# 
    return metadata_row
# 
def generate_csv(rootdir_itr):
    print(rootdir_itr)
    # 
    if rootdir_itr != "/home/teh_devs/deepfake/raw/combined_metadata.csv" and rootdir_itr != '/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv' and rootdir_itr != '/home/teh_devs/deepfake/raw/new_combined_metadata.csv':
        csv_rows = []
        with open(os.path.join(rootdir_itr, 'audios_metadata.csv'), "w") as empty_csv:
            pass
        # 
        real_audio = os.path.join(rootdir_itr, 'audio/real')
        fake_audio = os.path.join(rootdir_itr, 'audio/fake')
    # 
        for audio in os.listdir(real_audio):
            row = get_mediainfo_metadata(real_audio, audio)
            if row:
                csv_rows.append(row)
    # 
        for audio in os.listdir(fake_audio):
            row = get_mediainfo_metadata(fake_audio, audio)
            if row:
                csv_rows.append(row)
    # 
        with open(os.path.join(rootdir_itr, 'audios_metadata.csv'), 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_rows)
# 
if __name__ == "__main__":
    rootdir_itr = glob('/home/teh_devs/deepfake/raw/*')
    # rootdir_itr = rootdir_itr[:2]
    # print(rootdir_itr)
# 
    # generate_csv(rootdir_itr)
# 
    with ProcessPoolExecutor(max_workers=60) as executor:
        executor.map(generate_csv, rootdir_itr) 