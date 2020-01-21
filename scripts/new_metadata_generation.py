import os
import subprocess
import json
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from glob import glob
import numpy as np

rootdir = '/home/teh_devs/deepfake/raw'
csv_path = '/home/teh_devs/deepfake/deepfake-detection/video_metadata.csv'

csv_mediainfo_header = ["Complete name", "Format", "Format profile", "Codec ID", "File size", "Duration", "Overall bit rate mode", "Overall bit rate", "Writing application", "Video ID", "Video Format", "Video Format/Info", "Video Format profile", "Video Format settings", "Video Format settings, CABAC", "Video Format settings, ReFrames", "Video Codec ID", "Video Codec ID/Info", "Video Duration", "Video Bit rate", "Video Width", "Video Height", "Video Display aspect ratio", "Video Frame rate mode", "Video Frame rate", "Video Color space", "Video Chroma subsampling", "Video Bit depth", "Video Scan type", "Video Bits/(Pixel*Frame)", "Video Stream size", "Video Writing library", "Video Encoding settings", "Audio ID", "Audio Format", "Audio Format/Info", "Audio Format profile", "Audio Codec ID", "Audio Duration", "Audio Duration_LastFrame", "Audio Bit rate mode", "Audio Bit rate", "Audio Maximum bit rate", "Audio Channel(s)", "Audio Channel(s)_Original", "Audio Channel positions", "Audio Sampling rate", "Audio Frame rate", "Audio Compression mode", "Audio Stream size", "Audio Default", "Audio Alternate group"]

def get_statinfo_metadata(filename):
    statinfo_metadata = subprocess.check_output(['stat', filename])
    statinfo_metadata = statinfo_metadata.decode('utf8')

    statinfo_metadata = statinfo_metadata.split('Modify:')
    statinfo_metadata = statinfo_metadata[1].split('Change:')

    metadata_row = []

    for metadata in statinfo_metadata:
        metadata_info = metadata.split('\n')
        if metadata_info[0]:
            metadata_row.append(metadata_info[0].strip())
        else:
            metadata_row.append(np.nan)

    return metadata_row

def get_mediainfo_metadata(filename):
    mediainfo_metadata = subprocess.check_output(['mediainfo', '-json',filename])
    mediainfo_metadata = mediainfo_metadata.decode('utf8')
    
    mediainfo_metadata = mediainfo_metadata.split('\n')

    label = ""
    metadata_row = []
    metadata_present = {}

    for metadata in mediainfo_metadata:
        metainfo = metadata.split(':', 1)    
        metainfo = list(filter(None, metainfo)) 
        
        if len(metainfo) == 1:
            metainfo[0] = metainfo[0].strip()
            if metainfo[0] == "Video":
                label = "Video"
            elif metainfo[0] == "Audio":
                label = "Audio"
                    
        if len(metainfo) == 2:
            metainfo[0] = metainfo[0].strip()
            metainfo[1] = metainfo[1].strip()

            if label:
                key = label + " " + metainfo[0]
                metadata_present[key] = metainfo[1]
            else:
                metadata_present[metainfo[0]] = metainfo[1]

    for heading in csv_mediainfo_header:
        if metadata_present.get(heading):
            metadata_row.append(metadata_present[heading])
        else:
            metadata_row.append(np.nan)

    return metadata_row
            
csv_rows = []

def generate_csv(video):    
    
    if video.endswith(".mp4"):
        metadata_row = get_mediainfo_metadata(video)
        metadata_row.extend(get_statinfo_metadata(video))
        return metadata_row
   
    return None
    
if __name__ == "__main__":
    rootdir = glob('/home/teh_devs/deepfake/raw/dfdc_train_part_*')
    # rootdir = rootdir[:2]
    
    count = 1
    totallength = 0

    for folder in rootdir:
        with open(os.path.join(folder, 'videos_metadata.csv'), "w") as empty_csv:
            pass

        folders = glob(folder+ "/*")
        
        with ProcessPoolExecutor(max_workers=60) as executor:
            results = executor.map(generate_csv, folders)
        
        csv_rows = []

        for result in results:
            if result == None:
                continue
            csv_rows.append(result)
        
        totallength += len(csv_rows)

        print(count, folder, len(csv_rows), totallength)
        count+=1
            
        with open(os.path.join(folder, 'videos_metadata.csv'), "a") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_rows)
        

