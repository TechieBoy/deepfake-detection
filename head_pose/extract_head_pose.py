import numpy as np
from glob import glob
import os
from concurrent.futures import ProcessPoolExecutor
import json
from head_pose_estimator import HeadPoseEstimator
from face import face_68_landmarks
import cv2
from model_loader import get_points_from_landmarks, get_68_3d_model

dataset_folder_fake = '../dataset/fake/'
dataset_folder_real = '../dataset/real/'


def write_imgs_to_file(file_name, imgs):
    with open(file_name, 'ab') as bin_file:
        for img in imgs:
            feat = convert_image_to_head_pose_diff(img)
            if feat is None:
                continue
            np.savetxt(bin_file, feat, fmt='%.32f', delimiter=',', newline=' ')
            bin_file.write(os.linesep.encode('utf-8'))


def process_folder(f):
    print(f'Processing folder {f}')
    metafile = os.path.join(f, 'metadata.json')
    with open(metafile, 'r') as w:
        meta = json.load(w)
        for key, value in meta.items():
            folder_name = key.split('.')[0]
            vidFramesFolder = os.path.join(f, 'frames', folder_name)
            is_fake = True if value.get('label') == 'FAKE' else False
            imgs = glob(vidFramesFolder + "/*")
            folder = f.split('/')[-1]
            if is_fake:
                write_imgs_to_file(f'feat/fake_{folder}.txt', imgs)
            else:
                write_imgs_to_file(f'feat/real_{folder}.txt', imgs)


def convert_image_to_head_pose_diff(pic):
    pic = cv2.imread(pic)
    if pic is None:
        return None
    landmarks = face_68_landmarks(pic)
    height, width = pic.shape[:2]

    pose_estimator = HeadPoseEstimator(image_size=(height, width))
    try:
        marks = landmarks[0]
    except:
        return None
    image_points = get_points_from_landmarks(marks, "whole_face")
    ra, ta = pose_estimator.solve_pose("whole_face", image_points)

    image_points = get_points_from_landmarks(marks, "central_face")
    rc, tc = pose_estimator.solve_pose("central_face", image_points)
    r_diff = np.ravel(ra - rc)
    t_diff = np.ravel(ta - tc)

    feature = np.concatenate([r_diff, t_diff])
    # feat = np.ravel(scaler.fit_transform(feature.reshape(-1, 1)))
    return feature


def file_to_numpy_arrays(file_name):
    with open(file_name, 'r') as my_file:
        my_file = my_file.readlines()
        # Process pool here to read multiple arrays together
        for fk in my_file:
            a = np.fromstring(fk, dtype=np.float64, sep=' ')
            # Do something with a



if __name__ == '__main__':
    file_list = glob('/home/teh_devs/deepfake/raw/*')
    with ProcessPoolExecutor() as executor:
        executor.map(process_folder, file_list)
    # move_frames('/home/teh_devs/deepfake/raw/dfdc_train_part_0')