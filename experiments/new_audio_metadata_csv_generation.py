import csv
import re 
import pandas as pd

new_csv = '/home/teh_devs/deepfake/raw/audio_metadata.csv'
old_csv = '/home/teh_devs/deepfake/raw/combined_audioinfo_metadata.csv'

def generate_new_csv():
    rownumber = 1

    heading = "Folder_Number, Audio_Name, File_Size, Sampling_Rate, Frame_Rate, Stream_Size, Fake\n"

    with open(new_csv, 'w') as new_csv_file:
        new_csv_file.write(heading)

    with open(old_csv, 'r') as old_csv_file:
        contents = csv.reader(old_csv_file)

        for row in contents:
            if rownumber == 1:
                rownumber = 0
                continue

            print(row)

            new_row = []

            audio_name_details = row[0].rsplit('/', 1)

            audio_name = audio_name_details[1]

            folder_number = audio_name_details[0]

            temp = re.findall(r'\d+', folder_number) 
            res = list(map(int, temp)) 
            folder_number = res[0]

            if audio_name_details[0].rsplit('/', 1)[1] == 'real':
                fake = 0
            else:
                fake = 1
            
            file_size = row[3].split(" ")[1]
            
            sampling_rate = row[-4].split(" ")[1]
            
            frame_rate = row[-3].split(" ")[1]
            
            stream_size = row[-1].split(" ")[1]

            new_row.append(folder_number)
            new_row.append(audio_name)
            new_row.append(file_size)
            new_row.append(sampling_rate)
            new_row.append(frame_rate)
            new_row.append(stream_size)
            new_row.append(fake)

            with open(new_csv, 'a') as new_csv_file:
                writer = csv.writer(new_csv_file)
                writer.writerow(new_row)          

if __name__ == "__main__":
    generate_new_csv()