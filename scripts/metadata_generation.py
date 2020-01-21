import os
import subprocess
import json
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from glob import glob

rootdir = '/home/teh_devs/deepfake/raw'
csv_path = '/home/teh_devs/deepfake/deepfake-detection/video_metadata.csv'

def get_statinfo_metadata(filename):
    statinfo_metadata = subprocess.check_output(['stat', filename])
    statinfo_metadata = statinfo_metadata.decode('utf8')

    statinfo_metadata = statinfo_metadata.split('Modify:')

    statinfo_metadata = statinfo_metadata[1].split('Change:')

    metadata_row = []

    for metadata in statinfo_metadata:
        metadata_info = metadata.split('\n')
        metadata_row.append(metadata_info[0])

    return metadata_row

def get_mediainfo_metadata(filename):
    mediainfo_metadata = subprocess.check_output(['mediainfo', '-json',filename])
    
    mediainfo_metadata = mediainfo_metadata.decode('utf8')
    
    
    mediainfo_metadata = mediainfo_metadata.split('\n')
    print(len(mediainfo_metadata), mediainfo_metadata)

    metadata_row = []
    count = 1
    for metadata in mediainfo_metadata:
        metainfo = metadata.split(':', 1)
        metainfo = list(filter(None, metainfo)) 
        print(count, metainfo)
        count+=1
        
        if len(metainfo) == 2:
            metadata_row.append(metainfo[1])

    return metadata_row


            
def get_metadata(rootdir_itr, video):
    if video != "metadata.json" and video != "frames" and video != "audio" and video != "videos_metadata.csv":
        video = os.path.join(rootdir_itr, video)
        metadata_row = get_mediainfo_metadata(video)
        metadata_row.extend(get_statinfo_metadata(video))
        return metadata_row
    
    return None

def generate_csv(rootdir_itr):
    print(rootdir_itr)
    
    csv_rows = []

    with open(os.path.join(rootdir_itr, 'videos_metadata.csv'), "w") as empty_csv:
        pass 
 
    for video in os.listdir(rootdir_itr):
        if video != "combined_metadata.csv":
            # print(video)
            row = get_metadata(rootdir_itr, video)
            if row:
                csv_rows.append(row)
    
    with open(os.path.join(rootdir_itr, 'videos_metadata.csv'), 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)

if __name__ == "__main__":
    # rootdir_itr = glob('/home/teh_devs/deepfake/raw/*')
    # rootdir_itr = rootdir_itr[:2]
    # print(rootdir_itr)
    # with ProcessPoolExecutor(max_workers=60) as executor:
        # executor.map(generate_csv, rootdir_itr)

    filename = '/home/teh_devs/deepfake/raw/dfdc_train_part_48/qnvjmuhqif.mp4'

    mediainfo_metadata = get_mediainfo_metadata(filename)

    print(len(mediainfo_metadata))
  
