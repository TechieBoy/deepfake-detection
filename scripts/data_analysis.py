import os
from glob import glob

data = '/home/teh_devs/deepfake/output.txt'
output_file = '/home/teh_devs/deepfake/analysis.txt'

video_image_frame_count = {}

def videos_with_most_frames_missing(image_data):

    for line in image_data:
        name = "_".join(line.split("_", 2)[:2])
        
        if name in video_image_frame_count:
            video_image_frame_count[name] += 1
        else:
            video_image_frame_count[name] = 1

    videos = []

    for key, value in video_image_frame_count.items():
        if value > 7:
            # print(key)
            videos.append(key)
    
    return videos

def videos_with_all_frames_missing(videos_with_most_frames_missing_list):
    videos = []
    
    for video in videos_with_most_frames_missing_list:
        if video_image_frame_count[video] > 27:
            videos.append(video)
    
    return videos

def videos_with_no_frames(videos_with_no_frames_list):
    videos = []
    
    for video in videos_with_no_frames_list:
        if video_image_frame_count[video] == 30:
            videos.append(video)
    
    return videos

def find_full_path(filename):
    rootdir = glob('/home/teh_devs/deepfake/raw/*')

    for folder in rootdir:
        if not folder.endswith('.csv'):
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                return path
    return None

def find_stats():
    file = open(data)

    image_data = []

    for linenumber, line in enumerate(file):
        if linenumber == 468819:
            break
        else:
            image_data.append(line)

    file_obj = open(output_file, "w")

# -----------------------------------------------------------------------------------------------------------------------

    videos_with_most_frames_missing_list = videos_with_most_frames_missing(image_data)
    print("videos_with_most_frames_missing_list", len(videos_with_most_frames_missing_list))
    
    file_obj.write("Videos with most frames missing (>7)\n")
    for video in videos_with_most_frames_missing_list:
        file_obj.write(video)
        file_obj.write("\n")

# -----------------------------------------------------------------------------------------------------------------------

    videos_with_all_frames_missing_list = videos_with_all_frames_missing(videos_with_most_frames_missing_list)
    print("videos_with_all_frames_missing_list", len(videos_with_all_frames_missing_list))
    
    paths = []
    for video in videos_with_all_frames_missing_list:
        name = video.rsplit("/", 1)[1] + ".mp4"
        path = find_full_path(name)
        if path:
            paths.append(path)
        else:
            paths.append("Path not found" + name)

    file_obj.write("\nVideos with all frames missing (>27)\n")
    for video in paths:
        file_obj.write(video)
        file_obj.write("\n")

# -----------------------------------------------------------------------------------------------------------------------

    videos_with_no_frames_list = videos_with_no_frames(videos_with_all_frames_missing_list)
    print("videos_with_no_frames_list", len(videos_with_no_frames_list))
    
    paths = []
    for video in videos_with_no_frames_list:
        name = video.rsplit("/", 1)[1] + ".mp4"
        path = find_full_path(name)
        if path:
            paths.append(path)
        else:
            paths.append("Path not found" + name)

    file_obj.write("\nVideos with no frames (=30)\n")
    for video in paths:
        file_obj.write(video)
        file_obj.write("\n")


if __name__ == "__main__":
    find_stats()