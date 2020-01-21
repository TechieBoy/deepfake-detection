import os
from glob import glob
import csv
import shutil  

csv_path = "/home/teh_devs/deepfake/raw/iframes/real.csv"

if __name__ == "__main__":
    count = 0
    with open(os.path.join(csv_path), 'r') as real_videos:
        contents = real_videos.read()
        files = contents.split('\n')
        for i in files:
            # print(i)
            base = "/home/teh_devs/deepfake/raw/iframes/real/"
            rootdir = glob("/home/teh_devs/deepfake/raw/iframes/"+i+"*.jpg")
            # if j.split('/')[-1] not in ['real','real.csv']:
                # shutil.move(j,base + j.split('/')[-1])
            for j in rootdir:
                # print(j)
                count = count + 1

    print(count)
                

        # writer.writerows(csv_rows)
    
    # 
    # csv_rows = []

    # # with ProcessPoolExecutor(max_workers=60) as executor:
    # #     executor.map(generate_csv, rootdir_itr)
    # for folder in rootdir:
    #     if (
    #         folder != "/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv" and folder != "/home/teh_devs/deepfake/raw/combined_metadata.csv"

    #     ):
    #         rows = merge_csv(folder)
    #         # print(rows)
    #         if rows:
    #             for i in rows:
    #                 csv_rows.append(i)
    # # print(type(csv_rows))
    # with open(csv_path, 'w') as csvfile:
    #     print('Type',csv_rows)
    #     writer = csv.writer(csvfile)
    #     writer.writerows(csv_rows)