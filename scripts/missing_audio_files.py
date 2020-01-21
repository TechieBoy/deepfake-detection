import os
import csv

videos_metadata_csv = '/home/teh_devs/deepfake/raw/new_combined_metadata.csv'
audios_metadata_csv = '/home/teh_devs/deepfake/raw/combined_audioinfo_metadata.csv'

def check_missing_files():
    rownumber = 1

    videos = []
    audios = []

    with open(videos_metadata_csv, "r") as csv_file:
        video_contents = csv.reader(csv_file)

        for row in video_contents:
            if rownumber == 1:
                rownumber = 0
                continue

            path=row[-1]
            path+='/'
            name=row[0].split('.')[0]
            path+=name

            videos.append(path)

    rownumber = 1

    with open(audios_metadata_csv, "r") as csv_file:
        audio_contents = csv.reader(csv_file)

        for row in audio_contents:
            if rownumber == 1:
                rownumber = 0
                continue

            names=row[0].split('audio')
            path = names[0].split(' ')[1]

            if 'real' in names[1]:
                name = names[1].split('real/')[1]
            else:
                name = names[1].split('fake/')[1]
            name=name.split('.')[0]
            
            path+=name

            audios.append(path)       
 
    print(len(audios)) 
    print(len(videos))
 
    missing_audios = list(set(videos) - set(audios))

    print(len(missing_audios))
    for audio in missing_audios:
        print(audio)



if __name__ == "__main__":
    check_missing_files()