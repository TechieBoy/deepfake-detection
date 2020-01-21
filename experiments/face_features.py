import cv2
from skimage import feature
from skimage.transform import integral_image
from skimage.feature import haar_like_feature, local_binary_pattern

import numpy as np

import timeit

# OpenCV
# def get_HOG_features_using_CV(imagePath):
#     im = cv2.imread(imagePath)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     hog = cv2.HOGDescriptor()
#     H = hog.compute(im)
#     return H


# skImage
def get_HOG_features(imagePath, imageSize):
    im = read_gray_resize(imagePath, imageSize)
    # H = feature.hog(im, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(16, 16), transform_sqrt=True, block_norm="L1")
    H = feature.hog(im)
    return H


def get_viola_jones_features(imagePath, imageSize):
    im = read_gray_resize(imagePath, imageSize)
    ii = integral_image(im)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1])


def get_lbp_features(imagePath, imageSize):
    im = read_gray_resize(imagePath, imageSize)
    radius = 3
    n_points = 8 * radius
    METHOD = "uniform"
    return local_binary_pattern(im, n_points, radius, METHOD)


def read_gray_resize(imagePath, imageSize):
    im = cv2.imread(imagePath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, imageSize)
    return im


if __name__ == "__main__":

    sample = "../raw/dfdc_train_part_0/frames/aaqaifqrwn/aaqaifqrwn_face_0.jpg"
    imageSize = (256, 256)

    import time

    start = time.time()
    H = get_lbp_features(sample, imageSize)
    end = time.time()

    print(H.ravel())
    print(H)
    print(H.shape)
    print("Execution time:", end - start)
