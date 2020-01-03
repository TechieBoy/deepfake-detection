import os

import cv2
import numpy as np

from head_pose_estimator import HeadPoseEstimator
from face import face_68_landmarks
from model_loader import get_points_from_landmarks, get_68_3d_model
import glob

from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)


def sample_images(images_dir):
    files_extend = ["jpg", "jpeg", "gif", "png"]
    files = []
    for ext in files_extend:
        files += glob.glob("{}/*.{}".format(images_dir, ext))
    return files


def estimate(image_file, mode="nose_2eyes"):
    im = cv2.imread(image_file)
    landmarks = face_68_landmarks(im)
    height, width = im.shape[:2]
    print(height, width)

    pose_estimator = HeadPoseEstimator(image_size=(height, width), mode=mode)
    for marks in landmarks:
        # for pnt in marks:
        #     cv2.circle(im, (int(pnt[0]), int(pnt[1])), 1, (0, 255, 0), 2, cv2.LINE_AA)
        image_points = get_points_from_landmarks(marks, mode)
        print("========len======== : ", len(image_points))
        # print('========   ======== : ', image_points)
        rotation_vector, translation_vector = pose_estimator.solve_pose(image_points)
        print(
            "-------------R-----------\n",
            rotation_vector,
            "T||||\n",
            translation_vector,
            "\n-------------------------\n",
        )
        # end_points_2d = pose_estimator.projection(rotation_vector, translation_vector)
        # colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 125, 125)]
        # for i, pnt in enumerate(image_points.tolist()):
        #     cv2.circle(im, (int(pnt[0]), int(pnt[1])), 1, colors[i % 6], 3, cv2.LINE_AA)

        # end_points_2d = np.array(end_points_2d).astype(np.int).tolist()
        # cv2.line(im, tuple(end_points_2d[5]), tuple(end_points_2d[6]), (0, 255, 0))
        # cv2.line(im, tuple(end_points_2d[6]), tuple(end_points_2d[7]), (255, 0, 0))
        # cv2.line(im, tuple(end_points_2d[2]), tuple(end_points_2d[6]), (0, 0, 255))
    return rotation_vector, translation_vector
    # return im
    # cv2.imshow('im', im)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


OUTPUT = "output_nose_2eyes_02"

scaler = StandardScaler(copy=False)


def test(mode="nose_eyes_mouth"):
    image_files = sample_images(os.path.join(BASE_DIR, "sample_images"))
    print(image_files)
    output_dir = os.path.join(BASE_DIR, "{}".format(OUTPUT))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for im_f in image_files:
        f_name = im_f.split(os.sep)[-1]
        print("\n------------------------{}".format(f_name))
        im = estimate(im_f, mode)
        cv2.imwrite("{}/{}".format(output_dir, f_name), im)


def get_diff(mode1, mode2):
    image_files = sample_images(os.path.join(BASE_DIR, "sample_images"))
    print(image_files)
    for im_f in image_files:
        f_name = im_f.split(os.sep)[-1]
        print("\n------------------------{}".format(f_name))
        ra, ta = estimate(im_f, mode1)
        rc, tc = estimate(im_f, mode2)
        r_diff = np.ravel(ra - rc)
        t_diff = np.ravel(ta - tc)
        print("\n------------------------RA - RC ----------------------")
        print(r_diff)
        print("\n------------------------ta - tc ----------------------")
        print(t_diff)
        feat = np.concatenate([r_diff, t_diff])
        print(feat)
        print("\n-------------------------------------------")
        print(feat.reshape(1, -1))
        feat = np.ravel(scaler.fit_transform(feat.reshape(-1, 1)))
        print(feat)

        print(feat.shape)


# test('nose_eyes_ears')
# test('central_face')
# test('nose_chin_eyes_mouth')
# test('nose_eyes_mouth')
# test('nose_2eyes')
get_diff("nose_chin_eyes_mouth", "nose_eyes_ears")
# get_diff('whole_face', 'central_face')

