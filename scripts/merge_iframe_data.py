import os
from glob import glob
import csv

csv_path = "/home/teh_devs/deepfake/raw/combined_iframe_data.csv"


def merge_csv(folder):
    # print(folder.split('.')[-1])
    if len(folder.split('.')[-1]) > 5 and folder.split('/')[-1] != 'audio' and folder.split('/')[-1] != 'frames' and folder != 'csv':
        # print(folder)   
        with open(os.path.join(folder, (folder.split('/')[-1]+".txt")), "r") as csv_file:
            print((folder.split('/')[-1]))
            contents = csv_file.read()
            split_content = contents.split('\n')
            rows = []
            if split_content[-2].split(' ')[-2] == '29.97' or split_content[-2].split(' ')[-2] == '30':
                frame_rate = float(split_content[-2].split(' ')[-2])
            else:
                frame_rate = 30.00
            for i in range(len(split_content)-2):
                p_time = float(split_content[i].split(':')[-1])
                frame_num = int(p_time*frame_rate)
                row_contents = [folder+'.mp4',p_time,frame_num]
                rows.append(row_contents)
            return rows
    return []

        
            # with open(csv_path, "a") as combined_csv:
            #     combined_csv.write(contents)

def generate_csv(rootdir_itr):
    print(rootdir_itr)
    csv_rows = []
    # with open(os.path.join(rootdir_itr, 'videos_metadata.csv'), "w") as empty_csv:
    #     pass 
 
    for video in os.listdir(rootdir_itr):
        if video != "combined_metadata.csv":
            # print(video)
            row = get_metadata(rootdir_itr, video)
            if row:
                for i in row:
                    print(i)
                    csv_rows.append(i)
    
    with open(os.path.join(rootdir_itr, 'videos_metadata.csv'), 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)


if __name__ == "__main__":

    rootdir = glob("/home/teh_devs/deepfake/raw/dfdc_train_part_*/*")
    csv_rows = []

    # with ProcessPoolExecutor(max_workers=60) as executor:
    #     executor.map(generate_csv, rootdir_itr)
    for folder in rootdir:
        if (
            folder != "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv" and folder != "/home/teh_devs/deepfake/raw/combined_metadata.csv"

        ):
            rows = merge_csv(folder)
            # print(rows)
            if rows:
                for i in rows:
                    csv_rows.append(i)
    # print(type(csv_rows))
    with open(csv_path, 'w') as csvfile:
        print('Type',csv_rows)
        writer = csv.writer(csvfile)
        writer.writerows(csv_rows)
        
