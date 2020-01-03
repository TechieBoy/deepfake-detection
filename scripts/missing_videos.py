import os
import csv

csv_path = '/home/teh_devs/deepfake/raw/combined_metadata.csv'
new_csv = '/home/teh_devs/deepfake/raw/new_combined_metadata.csv'

def check_missing_videos():
    rownumber = 1

    with open(csv_path, 'r') as combined_metadata_csv:
        with open(new_csv, "a") as new_csv_file:
            contents = csv.reader(combined_metadata_csv)
            writer = csv.writer(new_csv_file)
    
            for row in contents:
                if rownumber == 1:
                    writer.writerow(row)
                    rownumber = 0
                    continue

                path = row[-1] 
                path +='/' 
                path +=row[0]

                if os.path.exists(path):        
                    writer.writerow(row)
                else:
                    print(row)


if __name__ == "__main__":
    check_missing_videos()