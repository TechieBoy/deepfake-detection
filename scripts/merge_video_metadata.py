import os
import pandas
from glob import glob

csv_path = "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv"

csv_mediainfo_header = "Complete name, Format, Format profile, Codec ID, File size, Duration, Overall bit rate mode, Overall bit rate, Writing application, Video ID, Video Format, Video Format/Info, Video Format profile, Video Format settings, Video Format settings CABAC, Video Format settings ReFrames, Video Codec ID, Video Codec ID/Info, Video Duration, Video Bit rate, Video Width, Video Height, Video Display aspect ratio, Video Frame rate mode, Video Frame rate, Video Color space, Video Chroma subsampling, Video Bit depth, Video Scan type, Video Bits/(Pixel*Frame), Video Stream size, Video Writing library, Video Encoding settings, Audio ID, Audio Format, Audio Format/Info, Audio Format profile, Audio Codec ID, Audio Duration, Audio Duration_LastFrame, Audio Bit rate mode, Audio Bit rate, Audio Maximum bit rate, Audio Channel(s), Audio Channel(s)_Original, Audio Channel positions, Audio Sampling rate, Audio Frame rate, Audio Compression mode, Audio Stream size, Audio Default, Audio Alternate group, Modify, Change\n"

def merge_csv(folder):
    # print(folder)
    with open(os.path.join(folder, "videos_metadata.csv"), "r") as csv_file:
        contents = csv_file.read()
        with open(csv_path, "a") as combined_csv:
            combined_csv.write(contents)


if __name__ == "__main__":
    rootdir = glob("/home/teh_devs/deepfake/raw/*")

    with open(csv_path, "w") as combined_csv:
        combined_csv.write(csv_mediainfo_header)
    
    for folder in rootdir:
        if not folder.endswith("iframes") and not folder.endswith(".csv"):
            merge_csv(folder)
