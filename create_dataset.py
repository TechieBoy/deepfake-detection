from glob import glob
import os
from concurrent.futures import ProcessPoolExecutor
import json
import shutil


dataset_folder_fake = '../dataset/fake/'
dataset_folder_real = '../dataset/real/'


def copy_all_files_within(src_folder, dest_folder):
    for child_file in os.listdir(src_folder):
        shutil.copy(os.path.join(src_folder, child_file), dest_folder)


def move_frames(f):
    print(f'Processing folder {f}')
    metafile = os.path.join(f, 'metadata.json')
    with open(metafile, 'r') as w:
        meta = json.load(w)
        for key, value in meta.items():
            vidFramesFolder = os.path.join(f, 'frames', key.split('.')[0])
            is_fake = True if value.get('label') == 'FAKE' else False

            if is_fake:
                copy_all_files_within(vidFramesFolder, dataset_folder_fake)
            else:
                copy_all_files_within(vidFramesFolder, dataset_folder_real)


if __name__=='__main__':
    folder_list = glob('../raw/*')
    # with ProcessPoolExecutor(max_workers=50) as executor:
    #     executor.map(move_frames, folder_list)