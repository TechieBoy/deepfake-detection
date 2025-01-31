import torch
import numpy as np
import cv2
from torchvision import transforms
from glob import glob
import csv
from meso import get_model
from facenet_pytorch import MTCNN
from PIL import Image
from scipy.stats.mstats import gmean
from torch.nn.functional import softmax

data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


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


def detect_faces_mtcnn(detector, frames):
    pil_images = [Image.fromarray(frame) for frame in frames]
    faces = detector(pil_images)
    return faces


def eval_model(device, class_mapping, model, faces):
    """Return geometric mean of frame detections"""

    fake_class = class_mapping["fake"]
    faces = [data_transform(f) for f in faces if f is not None]
    if not faces:
        return None
    faces = torch.stack(faces)
    d = faces.to(device)
    outputs = model(d)

    outputs = softmax(outputs, dim=1)
    probs = outputs[:, fake_class].cpu().detach().numpy()
    return gmean(probs)


def get_evenly_spaced_frames(cap, num_frames):
    total_frames = get_frame_count(cap)
    if num_frames == 1:
        return get_exact_frames(cap, [total_frames // 2])
    frame_indices = np.linspace(total_frames // 4, (3 * total_frames) // 4, num_frames, dtype=np.int)
    return get_exact_frames(cap, frame_indices)


if __name__ == "__main__":
    # dlib.DLIB_USE_CUDA
    class_mapping = {"fake": 0, "real": 1}
    model = get_model(2)
    device = torch.device("cuda:0")
    detector = MTCNN(image_size=300, margin=30, device=device, post_process=False)
    model.to(device)
    model.load_state_dict(torch.load("saved_models/meso.pt"))
    # Eval mode
    model.eval()

    fd = {}

    files = glob("../kaggle/test_videos/*")
    for fil in files:
        try:
            video_name = fil.split("/")[-1]
            print(video_name)
            cap = cv2.VideoCapture(fil)

            frames = get_evenly_spaced_frames(cap, 1)
            faces = detect_faces_mtcnn(detector, frames)

            sol = eval_model(device, class_mapping, model, faces)
            if sol is None:
                print("Frame list empty, trying with 2 frames next")

                frames = get_evenly_spaced_frames(cap, 2)
                faces = detect_faces_mtcnn(detector, frames)

                sol = eval_model(device, class_mapping, model, faces)
                if sol is None:
                    sol = 0.5
            cap.release()
        except Exception as e:
            print(e)
            sol = 0.5

        print(sol)
        fd[video_name] = sol
    with open("submission.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label"])
        for key, value in fd.items():
            writer.writerow([key, value])
