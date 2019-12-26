import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import cvlib as cv
from glob import glob
import os
import cv2
from concurrent.futures import ProcessPoolExecutor
from time import time

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


if __name__ == "__main__":
    st = time()
    with ProcessPoolExecutor() as executor:
        create_frames(executor)
    end = time()
    print("Took time " + str((end - st) / 60) + " minutes ")
    cv2.destroyAllWindows()
    # print(folder_list[:5])
