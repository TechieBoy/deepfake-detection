import csv

metadata_csv = '/home/teh_devs/deepfake/raw/combined_videoinfo_metadata.csv'
video_csv = '/home/teh_devs/deepfake/raw/new_combined_metadata.csv'

new_csv = '/home/teh_devs/deepfake/raw/video_metadata_analysis.csv'

mapping = {}

def read_labels():
    with open(video_csv, "r") as video_info_csv:
        contents = csv.reader(video_info_csv)

        for row in contents:
            if row[1] == "FAKE":
                mapping[row[0]] = 1
            else:
                mapping[row[0]] = 0


def add_labels_rows():
    with open(metadata_csv, "r") as metadata_csv_file:
        contents = csv.reader(metadata_csv_file)

        with open(new_csv, "a") as new_csv_file:
            writer = csv.writer(new_csv_file)
        
            rownum = 1

            for row in contents:
                if rownum == 1:
                    rownum = 0
                    new_row = row
                    print(new_row)
                    new_row.append("Fake")

                    writer.writerow(new_row)                

                    continue
                
                video_name_details = row[0].rsplit('/', 1)
                video_name = video_name_details[1]

                new_row = row 
                new_row.append(mapping[video_name])
                writer.writerow(new_row)

if __name__ == "__main__":
    with open(new_csv, "w") as new_csv_file:
        pass
    
    read_labels()

    add_labels_rows()