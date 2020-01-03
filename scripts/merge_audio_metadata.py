import os
from glob import glob

csv_path = "/home/teh_devs/deepfake/raw/combined_audioinfo_metadata.csv"

def merge_csv(folder):
    # print(folder)
    with open(os.path.join(folder, "audios_metadata.csv"), "r") as csv_file:
        contents = csv_file.read()
        with open(csv_path, "a") as combined_csv:
            combined_csv.write(contents)


if __name__ == "__main__":
    rootdir = glob("/home/teh_devs/deepfake/raw/*")

    heading = "Complete name, Format, Format/Info, File size, Overall bit rate mode, Audio Format, Format/Info, Format version, Format profile, Bit rate mode, Channel(s), Channel positions, Sampling rate, Frame rate, Compression mode, Stream size"

    with open(csv_path, "w") as combined_csv:
        combined_csv.write(heading)

    for folder in rootdir:
        if (
            folder != "/home/teh_devs/deepfake/raw/combined_audioinfo_metadata.csv"
            and folder != "/home/teh_devs/deepfake/raw/combined_metadata.csv" and folder != "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv" and folder != "/home/teh_devs/deepfake/raw/new_combined_metadata.csv"
        ):
            merge_csv(folder)