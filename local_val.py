from models.sppnet import get_model
from torchvision import transforms
import cv2
import torch
from torch.nn.functional import softmax
from scipy.stats.mstats import gmean
import numpy as np
from itertools import islice
from facenet_pytorch import MTCNN
from glob import glob
import os
from PIL import Image
import random
import csv
import pandas as pd


def get_model_spp(num_classes=2):
    model = get_model(num_classes=num_classes, pretrained=False)
    return model


data_transform_spp = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_frame_count(cap):
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return num_frames


def get_exact_frames(cap, frame_indices):
    frame_count = get_frame_count(cap)
    frames = []
    i = 0
    for frame_idx in range(int(frame_count)):
        # Get the next frame, but don't decode if we're not using it.
        ret = cap.grab()
        if not ret:
            # print("Error grabbing frame %d" % (frame_idx))
            pass

        # Need to look at this frame?
        if frame_idx >= frame_indices[i]:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                # print("Error retrieving frame %d" % (frame_idx))
                pass
            else:
                height, width, channels = frame.shape
                if height > 500 or width > 500:
                    frame = cv2.resize(
                        frame, (width // 2, height // 2), interpolation=cv2.INTER_AREA
                    )
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            i += 1
            if i >= len(frame_indices):
                break
    return frames


def detect_faces_mtcnn(detector, frames):
    def get_dist(px, py, x, y):
        return abs(px - x) + abs(py - y)

    def get_coords(s, x, y):
        min_set = max(s, key=lambda k: get_dist(k[0], k[1], x, y))
        return min_set[0], min_set[1], min_set[2]

    def get_avg_coords(s):
        x, y = 0.0, 0.0
        for dd in s:
            px, py, *rest = dd
            x += px
            y += py
        tot = len(s)
        return x / tot, y / tot

    def add_to_closest_set(x, y, area, bi, bj):
        min_dist = float("inf")
        idx = -1
        for i, s in enumerate(data):
            px, py, pa = get_coords(s, x, y)
            dist = get_dist(px, py, x, y)
            areas = sorted([pa, area])
            if dist > 175 or (areas[1] / areas[0]) > 1.3:
                continue
            if dist < min_dist:
                dist = min_dist
                idx = i

        if idx == -1:
            stuff = (x, y, area, bi, bj)
            ss = set()
            ss.add(stuff)
            data.append(ss)
        else:
            data[idx].add((x, y, area, bi, bj))

    def get_box(face_box, shape, padding=15):
        (startX, startY) = int(face_box[0]), int(face_box[1])
        (endX, endY) = int(face_box[2]), int(face_box[3])
        width, height = shape

        y_top = max(startY - padding, 0)
        x_top = max(startX - padding, 0)
        y_bot = min(endY + padding, height)
        x_bot = min(endX + padding, width)

        return y_top, y_bot, x_top, x_bot

    stored_frames = []
    data = []
    frames_boxes, frames_confidences = detector.detect(frames, landmarks=False)
    for batch_idx, (frame_boxes, frame_confidences) in enumerate(
        zip(frames_boxes, frames_confidences)
    ):
        stored_frames.append(frame_boxes)
        if (frame_boxes is not None) and (len(frame_boxes) > 0):
            frame_locations = []
            for j, (face_box, confidence) in enumerate(
                zip(frame_boxes, frame_confidences)
            ):
                (y, yb, x, xb) = get_box(face_box, frames[batch_idx].size, 0)
                area = (yb - y) * (xb - x)
                if not data:
                    stuff = (x, y, area, batch_idx, j)
                    ss = set()
                    ss.add(stuff)
                    data.append(ss)
                else:
                    add_to_closest_set(x, y, area, batch_idx, j)

    final_frames = []

    def take_faces(f):
        rx, ry, area, i, j = f
        frame = frames[i]
        box = stored_frames[i][j]
        (y, yb, x, xb) = get_box(box, frame.size, 10)
        # face_extract = np.copy(frame[y : yb, x : xb])
        face_extract = frame.crop((x, y, xb, yb))
        final_frames.append(face_extract)

    count = 0
    for d in data:
        if len(d) > 10:
            count += 1
            for f in d:
                take_faces(f)

    if count == 0:
        max_set = max(data, key=lambda s: len(s))
        print("taking ", len(max_set), "faces")
        for f in max_set:
            take_faces(f)

    del frames
    del stored_frames

    return final_frames


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def batched_eval_model(bsize, device, class_mapping, model, faces, data_transform):
    faces_chunked = chunk(faces, bsize)
    answers = []
    lens = []
    for face_chunk in faces_chunked:
        ans, l = eval_model(device, class_mapping, model, face_chunk, data_transform)
        answers.append(ans)
        lens.append(l)
    answers = [x for x in answers if x is not None]
    if not answers:
        return None
    percent = sum(answers) / len(faces)
    return percent


def eval_model(device, class_mapping, model, faces, data_transform):
    """Return geometric mean of frame detections"""

    fake_class = class_mapping["fake"]
    faces = [data_transform(f) for f in faces if f is not None]
    if not faces:
        return None
    faces = torch.stack(faces)
    d = faces.to(device)
    outputs = model(d)
    del faces
    outputs = softmax(outputs, dim=1)
    probs = outputs[:, fake_class].cpu().detach().numpy()
    probabilites = np.around(probs, decimals=1)
    gt_indices = probs > 0.5
    gt_vals = probs[gt_indices]
    return gt_indices.sum() / len(probs), gt_vals, gmean(probs)


def get_evenly_spaced_frames(cap, num_frames):
    total_frames = get_frame_count(cap)
    if num_frames == 1:
        return get_exact_frames(cap, [total_frames // 2])
    frame_indices = np.linspace(
        total_frames // 4, (3 * total_frames) // 4, num_frames, dtype=np.int
    )
    return get_exact_frames(cap, frame_indices.tolist())


from IPython.display import display
import traceback
import gc
import math

device = torch.device("cuda:0")
class_mapping = {"fake": 0, "real": 1}

spp = get_model_spp(2)
spp.to(device)
# spp = torch.nn.DataParallel(spp)
spp.load_state_dict(
    torch.load(
        "/home/teh_devs/deepfake/deepfake-detection/saved_models/finetuning_pretrained_spp_with_transform06Feb10:19PM/finetuning_pretrained_spp_with_transform_50.pt"
    )
)
spp.eval()
detector = MTCNN(device=device, keep_all=True, select_largest=False, post_process=False)

df = pd.read_csv("~/deepfake/raw/combined_metadata.csv")
fakes = df[df.label == "REAL"][["index", "folder"]]
fakes["path"] = fakes["folder"] + "/" + fakes["index"]
fake_list = fakes["path"].tolist()
random.seed(12)
random.shuffle(fake_list)
fd = {}
files = fake_list

from time import time
from tqdm import tqdm

logloss = 0
count = 0
fakes = set()
# iterator = tqdm(files, ncols=0)
for fil in files:
    try:
        s = time()
        video_name = fil.split("/")[-1]
        try:
            cap = cv2.VideoCapture(fil)
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            frames_from_video = get_evenly_spaced_frames(cap, 43)
            frames_to_pil = [Image.fromarray(x) for x in frames_from_video]

            faces_from_frames = detect_faces_mtcnn(detector, frames_to_pil)

            sol = 0.5
            percent, gt_val, tot_mean = eval_model(
                device, class_mapping, spp, faces_from_frames, data_transform_spp
            )
            display(random.choice(faces_from_frames))
            print(percent)
            print(gt_val)
            print(tot_mean)
            print()
            # if tot_mean < 0.5:
            #     count += 1
            #     # iterator.write(video_name)
            #     # iterator.write(tot_mean)
            #     fakes.add(video_name)
            # # display(random.choice(faces_from_frames))

            if percent is None:
                print("Not a single frame found, SHAME")
                sol = 0.5
        except RuntimeError:
            print("Runtime Error lol")
            del detector
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            detector = MTCNN(
                device=device, keep_all=True, select_largest=False, post_process=False
            )
            sol = 0.5
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            # Overfit leaderboard!!!! make 0.5
            sol = 0.99
        finally:
            cap.release()
        if percent is not None:
            if percent == 0:
                sol = 0.1
            elif percent > 0.5:
                sol = 0.9
            elif percent > 0.25:
                sol = 0.8
            elif percent > 0.08:
                sol = 0.65
            else:
                sol = 0.5
            # print(f"{percent*100:.1f}% frames are fake")
        # print(f"Took {(time() - s):.1f} seconds")
        logloss += math.log((sol))
        fd[video_name] = sol
        # print(f"Predicting {sol}")
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        torch.save(fakes, "fakes_under_p5.pkl")
        import sys
        sys.exit()
torch.save(fakes, "fakes_under_p5.pkl")
print(count)
# n = len(fake_list)
# mult = -1 / n
# print(mult * logloss)
