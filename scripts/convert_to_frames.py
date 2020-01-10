import os
import cv2
from concurrent.futures import ProcessPoolExecutor
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm
from PIL import Image
import pickle


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


def convert_video_to_frames_per_frame(capture, per_n):
    num_frames = get_frame_count(capture)
    frames = []
    for i in range(0, num_frames):
        ret = capture.grab()
        if i % per_n == 0:
            ret, frame = capture.retrieve()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = image.shape
                image = cv2.resize(
                    image, (width // 2, height // 2), interpolation=cv2.INTER_AREA
                )
                frames.append(frame)
    return frames


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
                image = cv2.resize(
                    image, (width // 2, height // 2), interpolation=cv2.INTER_AREA
                )
                frames.append(frame)
                index_list.append(idx)
    return frames, index_list


def load_model(device):
    device = torch.device(device)
    detector = MTCNN(image_size=300, margin=30, device=device, post_process=False)
    return detector


def detect_faces_mtcnn_and_save(
    detector, base_folder, base_video, frames, filenames=None
):
    pil_images = [Image.fromarray(frame) for frame in frames]
    if filenames is None:
        filenames = [
            os.path.join(base_folder, f"{base_video}_face_{i}.jpg")
            for i, _ in enumerate(pil_images)
        ]
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
                begin_indices = [
                    i for i in range(frame_begin, frame_end, total_frames // 4)
                ]

                frames, indices = get_exact_frames_for_optical_flow(
                    capture, begin_indices
                )
                
                new_video_folder = os.path.join(base_folder, name)
                os.mkdir(new_video_folder)
                filenames = [
                    os.path.join(new_video_folder, f"{name}_face_{i}.jpg")
                    for i in indices
                ]
                detect_faces_mtcnn_and_save(detector, new_video_folder, name, frames, filenames)
                capture.release()
            except Exception as e:
                print(video)
                print(e)
                continue


if __name__ == "__main__":
    base_folder = "/data/of/"

    folder_list = []
    print("Doing first 5 folders")
    for i in range(5):
        folder_list.append(f"/home/teh_devs/deepfake/raw/dfdc_train_part_{i}")

    detector = load_model(device="cuda:0")
    for f in folder_list:
        convert_video_to_frames_with_mtcnn(detector, base_folder, f)
