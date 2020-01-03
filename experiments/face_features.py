import cv2
from skimage import feature
from skimage.transform import integral_image
from skimage.feature import haar_like_feature

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
def get_HOG_features(imagePath):
    im = cv2.imread(imagePath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # H = feature.hog(im, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(16, 16), transform_sqrt=True, block_norm="L1")
    H = feature.hog(im)
    return H

def get_viola_jones_features(imagePath):
    im = cv2.imread(imagePath)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, (30, 30))
    ii = integral_image(im)
    return haar_like_feature(ii, 0, 0, ii.shape[0], ii.shape[1])


if __name__ == "__main__":

    sample = "../raw/dfdc_train_part_0/frames/aaqaifqrwn/aaqaifqrwn_face_0.jpg"

    start = timeit.timeit()
    H = get_viola_jones_features(sample)
    end = timeit.timeit()
    print(H)
    print(H.shape)
    print("Viola Jones", end - start)