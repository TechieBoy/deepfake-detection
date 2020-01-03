import os
from glob import glob

csv_path = "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv"


def merge_csv(folder):
    # print(folder)
    with open(os.path.join(folder, "videos_metadata.csv"), "r") as csv_file:
        contents = csv_file.read()
        with open(csv_path, "a") as combined_csv:
            combined_csv.write(contents)


if __name__ == "__main__":
    rootdir = glob("/home/teh_devs/deepfake/raw/*")
    for folder in rootdir:
        if (
            folder != "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv"
            and folder != "/home/teh_devs/deepfake/raw/combined_metadata.csv"
        ):
            merge_csv(folder)
