import os
import cv2
from concurrent.futures import ProcessPoolExecutor
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image
import pickle
from face_detection import RetinaFace
from bisect import bisect_left
from collections import Counter
import math

def delete_folders():
    """Deletes the frames folder from each directory in folder_list"""
    from shutil import rmtree

    for f in folder_list:
        folder_to_delete = os.path.join(f, "frames")
        rmtree(folder_to_delete)


def create_folders():
    """
    Creates a folder called frames in each directory and creates subfolders for
    each video in the frames folder.
    """
    for f in folder_list:
        os.mkdir(os.path.join(f, "frames"))
        for fil in os.listdir(f):
            fil = fil.split(".")[0]
            if fil != "metadata" and fil != "frames":
                os.mkdir(os.path.join(f, "frames", fil))


def convert_video_to_frames(input_path, output_folder):
    """Extract all frames from a video"""
    count = 0
    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.png"), frame)
        count += 1
    cap.release()


def find_max_face(input_image):
    """
    Finds face in input_image with maximum confidence and returns it
    Adds padding of 15px around face
    """
    detection = cv.detect_face(input_image)
    if detection is not None:
        faces, confidences = detection

        if confidences:
            max_conf = max(confidences)
            face = faces[confidences.index(max_conf)]

            (startX, startY) = face[0], face[1]
            (endX, endY) = face[2], face[3]
            height, width, _ = input_image.shape

            y_top = max(startY - 15, 0)
            x_top = max(startX - 15, 0)
            y_bot = min(endY + 15, height)
            x_bot = min(endX + 15, width)

            return input_image[y_top:y_bot, x_top:x_bot]
    return None


def convert_video_to_frames_periodic(name_prefix, input_path, output_folder, dt):
    """Captures a frame every dt milliseconds"""
    count = 0
    cap = cv2.VideoCapture(input_path)
    success, image = cap.read()
    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, (count * dt))
        success, frame = cap.read()
        cv2.imwrite(os.path.join(output_folder, f"{name_prefix}_frame_{count}.png"), frame)
        count += 1
    cap.release()


def convert_video_to_face_frames_periodic(name_prefix, input_path, output_folder, dt):
    """Captures a frame and tries to detect and save a face in it every dt milliseconds"""
    count = 0
    num_face = 0
    cap = cv2.VideoCapture(input_path)
    success, image = cap.read()
    while success:
        cap.set(cv2.CAP_PROP_POS_MSEC, (count * dt))
        success, frame = cap.read()
        face = find_max_face(frame)
        if face is not None:
            cv2.imwrite(os.path.join(output_folder, f"{name_prefix}_face_{num_face}.png"), face)
            num_face += 1
        count += 1
    if num_face < 5:
        print(name_prefix + f" has {num_face} faces")
    cap.release()


def create_frames(executor):
    for f in folder_list:
        print(f"In folder {f}")
        for video in os.listdir(f):
            if video != "metadata.json" and video != "frames":
                # print(f"Processing video {video}")
                input_path = os.path.join(f, video)
                video_folder = video.split(".")[0]
                output_folder = os.path.join(f, "frames", video_folder)
                executor.submit(convert_video_to_face_frames_periodic, video_folder, input_path, output_folder, 1000)
                # convert_video_to_face_frames_periodic(video_folder, input_path, output_folder, 800)


def convert_with_mtcnn_parallel(detector, base_folder, folder):
    print(folder)

    def func(video):
        return convert_video_to_frames_per_frame(os.path.join(folder, video), 10)

    video_list = os.listdir(folder)
    video_list.remove("metadata.json")
    video_list.remove("frames")
    video_list.remove("audio")
    with ProcessPoolExecutor(20) as pool:
        frame_list = pool.map(func, video_list, chunksize=1)
    for video, frames in zip(video_list, frame_list):
        base_video = video.split(".")[0]
        detect_faces_mtcnn_and_save(detector, base_folder, base_video, frames)



def get_frame_count(cap):
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames


def get_exact_frames(cap, frame_indices):
    """Gets all frames with the indices in frame indices (0 based)"""
    frames = []
    for index in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    return frames


def get_exact_frames_for_optical_flow(cap, frame_indices):
    """Gets all frames and 4 ahead with the indices in frame indices (0 based)"""
    frames = []
    index_list = []
    for index in frame_indices:
        for i in range(4):
            idx = index + i

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = image.shape
                image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                frames.append(image)
                index_list.append(idx)
    return frames, index_list


def load_model(device):
    device = torch.device(device)
    detector = MTCNN(device=device, keep_all=True, select_largest=False, post_process=False)
    return detector


def mtcnn_detect(detector, frames, path, vid_name):
    data = []
    def get_dist(px,py,x,y):
        return abs(px - x) + abs(py - y)
    
    def get_min_coords(s, x,y):
        min_set = max(s, key=lambda k:get_dist(k[0], k[1], x,y))
        return min_set[0], min_set[1], min_set[2]

    def get_avg_coords(s):
        x,y = 0.0,0.0
        for dd in s:
            px,py,*rest = dd
            x += px
            y += py
        tot = len(s)
        return x/tot, y/tot

    def add_to_closest_set(x,y,area,bi,bj):
        min_dist = float('inf')
        idx = -1
        for i, s in enumerate(data):
            px,py,pa = get_min_coords(s,x,y)
            dist = get_dist(px,py,x,y)
            areas = sorted([pa, area])
            if dist > 175 or (areas[1] / areas[0]) > 1.3:
                continue
            if dist < min_dist:
                dist = min_dist
                idx = i

        if idx == -1:
            stuff = (x,y,area,bi,bj,)
            ss = set()
            ss.add(stuff)
            data.append(ss)
        else:
            data[idx].add((x,y,area,bi,bj,))


    stored_frames = []
    def get_box(face_box, shape, padding=15):
        (startX, startY) = int(face_box[0]), int(face_box[1])
        (endX, endY) = int(face_box[2]), int(face_box[3])
        height, width, _ = shape

        y_top = max(startY - padding, 0)
        x_top = max(startX - padding, 0)
        y_bot = min(endY + padding, height)
        x_bot = min(endX + padding, width)

        return y_top, y_bot, x_top, x_bot
    
    frames_boxes, frames_confidences = detector.detect([Image.fromarray(x) for x in frames], landmarks=False)
    for batch_idx, (frame_boxes, frame_confidences) in enumerate(zip(frames_boxes, frames_confidences)):
        frame = frames[batch_idx]
        stored_frames.append(frame_boxes)
        if (frame_boxes is not None) and (len(frame_boxes) > 0):
            frame_locations = []
            for j, (face_box, confidence) in enumerate(zip(frame_boxes, frame_confidences)):
                (y, yb, x, xb) = get_box(face_box, frame.shape, 0)
                area = (yb - y) * (xb - x)
                if not data:
                    stuff = (x,y,area,batch_idx,j,)
                    ss = set()
                    ss.add(stuff)
                    data.append(ss)
                else:
                    add_to_closest_set(x,y,area,batch_idx,j)

    count = 0
    for i, d in enumerate(data):
        if len(d) > 9:
            for f in d:
                rx,ry,area,i,j = f
                frame = frames[i]
                box = stored_frames[i][j]
                (y, yb, x, xb) = get_box(box, frame.shape, 10)
                face_extract = frame[y : yb, x : xb]
                pa = f'{path}/{vid_name}_{len(d)}_{count}.png'
                cv2.imwrite(pa,cv2.cvtColor(face_extract, cv2.COLOR_RGB2BGR))
                count += 1


def convert_video_to_frames_per_frame(capture, per_n):
    num_frames = get_frame_count(capture)
    frames = []
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % per_n == 0:
            ret, image = capture.retrieve()
            if ret:
                height, width, channels = image.shape
                image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                frames.append(image)
    return frames

def load_model_retina(device):
    return RetinaFace(gpu_id=0)


def detect_faces_mtcnn_and_save(detector, base_folder, base_video, frames, filenames=None):
    pil_images = [Image.fromarray(frame) for frame in frames]
    if filenames is None:
        filenames = [os.path.join(base_folder, f"{base_video}_face_{i}.png") for i, _ in enumerate(pil_images)]
    faces = detector(pil_images, filenames)
    return faces


def convert_video_to_frames_with_mtcnn(detector, base_folder, folder):
    print(folder)
    for video in tqdm(os.listdir(folder)):
        name = video.split(".")
        try:
            name, extension = name[0], name[1]
        except IndexError:
            continue
        if extension == "mp4":
            try:
                capture = cv2.VideoCapture(os.path.join(folder, video))
                total_frames = get_frame_count(capture)
                frame_begin = 10
                frame_end = total_frames - 8
                begin_indices = [i for i in range(frame_begin, frame_end, total_frames // 4)]

                frames, indices = get_exact_frames_for_optical_flow(capture, begin_indices)

                new_video_folder = os.path.join(base_folder, name)
                os.mkdir(new_video_folder)
                filenames = [os.path.join(new_video_folder, f"{name}_face_{i}.png") for i in indices]
                detect_faces_mtcnn_and_save(detector, new_video_folder, name, frames, filenames)
                capture.release()
            except Exception as e:
                print(video)
                print(e)
                continue


if __name__ == "__main__":
    # base_folder = "/home/teh_devs/deepfake/raw/test_vids"
    """
    Rescaled by 4 need testing
    """
    from glob import glob
    storage_dir = '/home/teh_devs/deepfake/dataset/revamp'
    folder_list = []
    print("Doing first 5 folders")
    for i in range(0, 5):
        folder_list.append(f"/home/teh_devs/deepfake/raw/dfdc_train_part_{i}")
    
    detector = load_model(device="cuda:0")
    # f = '/home/teh_devs/deepfake/raw/dfdc_train_part_4/srqogltgnx.mp4'
    for f in folder_list:
        print(f)
        videos = glob(f + '/*.mp4')
        for vid in tqdm(videos, ncols=0):
            try:
                vid_name = vid.split('/')[-1].split('.')[0]
                capture = cv2.VideoCapture(vid)
                frames = convert_video_to_frames_per_frame(capture, 10)
                new_folder = os.path.join(storage_dir, vid_name)
                os.mkdir(new_folder)
                mtcnn_detect(detector, frames, new_folder, vid_name)
                capture.release()
            except Exception as e:
                print(e)
    # for f in folder_list:
    #     convert_video_to_frames_with_mtcnn(detector, base_folder, f)
