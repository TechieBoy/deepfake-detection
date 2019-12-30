from glob import glob
import os
import cv2
from concurrent.futures import ProcessPoolExecutor
from time import time
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image
import pandas as pd
from torchvision.utils import save_image
import numpy as np
folder_list = glob("../raw/*")


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
        if ret == False:
            break
        cv2.imwrite(os.path.join(output_folder, f"frame_{count}.jpg"), frame)
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
        cv2.imwrite(
            os.path.join(output_folder, f"{name_prefix}_frame_{count}.jpg"), frame
        )
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
            cv2.imwrite(
                os.path.join(output_folder, f"{name_prefix}_face_{num_face}.jpg"), face
            )
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
                executor.submit(
                    convert_video_to_face_frames_periodic,
                    video_folder,
                    input_path,
                    output_folder,
                    1000,
                )
                # convert_video_to_face_frames_periodic(video_folder, input_path, output_folder, 800)


def convert_video_to_frames_per_frame(input_path, per_n):
    capture = cv2.VideoCapture(input_path)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % per_n == 0:
            ret, frame = capture.retrieve()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = image.shape
                image = cv2.resize(image, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
                frames.append(frame)
    capture.release()
    return frames


def load_model():
    device = torch.device("cuda:0")
    detector = MTCNN(image_size=300, margin=30, device=device, post_process=False)
    return detector


def detect_faces_mtcnn_and_save(detector, base_folder, base_video, frames):
    pil_images = [Image.fromarray(frame) for frame in frames]
    filenames = [os.path.join(base_folder, f'{base_video}_face_{i}.jpg') for i, _ in enumerate(pil_images)]
    faces = detector(pil_images, filenames)
    return faces


def convert_video_to_frames_with_mtcnn(detector, base_folder, folder):
    print(folder)
    for video in tqdm(os.listdir(folder)):
        if video != "metadata.json" and video != "frames" and video != "audio":
            frames = convert_video_to_frames_per_frame(os.path.join(folder, video), 10)
            base_video = video.split('.')[0]
            detect_faces_mtcnn_and_save(detector, base_folder, base_video, frames)
            # for i, face in enumerate(faces):
            #     if type(face) == torch.Tensor:
            #         filename = os.path.join(base_folder, f'{base_video}_face_{i}.jpg')
            #         save_image(face, filename)


def convert_with_mtcnn_parallel(detector, base_folder, folder):
    print(folder)

    def func(video):
        return convert_video_to_frames_per_frame(os.path.join(folder, video), 10)

    video_list = os.listdir(folder)
    video_list.remove('metadata.json')
    video_list.remove('frames')
    video_list.remove('audio')
    with ProcessPoolExecutor(20) as pool:
        frame_list = pool.map(func, video_list, chunksize=1)
    for video, frames in zip(video_list, frame_list):
        base_video = video.split('.')[0]
        detect_faces_mtcnn_and_save(detector, base_folder, base_video, frames)
        

if __name__ == "__main__":
    detector = load_model()
    base_folder = '../dataset/new/'
    for f in folder_list:
        convert_video_to_frames_with_mtcnn(detector, base_folder, f)
    # def multiconvert(folder):
    #     if folder != '../raw/combined_metadata.csv':
    #         return convert_video_to_frames_with_mtcnn(detector, base_folder, folder)

    # for f in reversed(folder_list):
    #     multiconvert(f)
    # for f in folder_list[:1]:
    #     convert_video_to_frames_with_mtcnn(base_folder, f)
    # st = time()
    # import functools
    # ff = functools.partial(convert_video_to_frames_with_mtcnn, detector=None, base_folder=base_folder)
    # print(ff)
    # with ProcessPoolExecutor(50) as executor:
    #     print("HERE")
    #     executor.map(save_to_numpy, folder_list[1:])
    # end = time()
    # print("Took time " + str((end - st) / 60) + " minutes ")
    # cv2.destroyAllWindows()
    # print(folder_list[:5])
