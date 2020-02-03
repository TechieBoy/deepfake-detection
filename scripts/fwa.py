import random
import math
import pandas as pd
import os
import cv2
import numpy as np
import dlib
import albumentations as A
from glob import glob
from tqdm import tqdm

def get_array_in_batch(arr, shuffle=False, seed=50, per=5000):
    if shuffle:
        random.seed(seed)
        random.shuffle(arr)
    div = math.ceil(len(arr) / per)
    batched = []
    for i in range(div):
        batched.append(arr[i * per : (i + 1) * per])
    return div, batched


def get_split_df(seed=50, per=5000):
    df = pd.read_csv("/home/teh_devs/deepfake/dataset/fake-real-distinct.csv")
    df = df[
        ((df.video_label == "FAKE") | (df.video_label == "REAL"))
        & (df.audio_label == "REAL")
    ]
    dff = df[(df.video_label == "FAKE")]
    df_reals = df[(df.video_label == "REAL")].filename.to_list()
    div, reals = get_array_in_batch(df_reals, shuffle=True, seed=seed, per=per)
    fakes = [[] for _ in range(div)]

    grouped = dff.groupby(dff.original)
    removed = []
    for i, rr in enumerate(reals):
        for r in rr:
            try:
                fakes[i].extend(grouped.get_group(r).filename.to_list())
            except KeyError:
                removed.append(r)
    return reals, fakes


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:

        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


mean_face_x = np.array(
    [
        0.000213256,
        0.0752622,
        0.18113,
        0.29077,
        0.393397,
        0.586856,
        0.689483,
        0.799124,
        0.904991,
        0.98004,
        0.490127,
        0.490127,
        0.490127,
        0.490127,
        0.36688,
        0.426036,
        0.490127,
        0.554217,
        0.613373,
        0.121737,
        0.187122,
        0.265825,
        0.334606,
        0.260918,
        0.182743,
        0.645647,
        0.714428,
        0.793132,
        0.858516,
        0.79751,
        0.719335,
        0.254149,
        0.340985,
        0.428858,
        0.490127,
        0.551395,
        0.639268,
        0.726104,
        0.642159,
        0.556721,
        0.490127,
        0.423532,
        0.338094,
        0.290379,
        0.428096,
        0.490127,
        0.552157,
        0.689874,
        0.553364,
        0.490127,
        0.42689,
    ]
)

mean_face_y = np.array(
    [
        0.106454,
        0.038915,
        0.0187482,
        0.0344891,
        0.0773906,
        0.0773906,
        0.0344891,
        0.0187482,
        0.038915,
        0.106454,
        0.203352,
        0.307009,
        0.409805,
        0.515625,
        0.587326,
        0.609345,
        0.628106,
        0.609345,
        0.587326,
        0.216423,
        0.178758,
        0.179852,
        0.231733,
        0.245099,
        0.244077,
        0.231733,
        0.179852,
        0.178758,
        0.216423,
        0.244077,
        0.245099,
        0.780233,
        0.745405,
        0.727388,
        0.742578,
        0.727388,
        0.745405,
        0.780233,
        0.864805,
        0.902192,
        0.909281,
        0.902192,
        0.864805,
        0.784792,
        0.778746,
        0.785343,
        0.778746,
        0.784792,
        0.824182,
        0.831803,
        0.824182,
    ]
)

landmarks_2D = np.stack([mean_face_x, mean_face_y], axis=1)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (
    LEFT_BROW_POINTS
    + RIGHT_EYE_POINTS
    + LEFT_EYE_POINTS
    + RIGHT_BROW_POINTS
    + NOSE_POINTS
    + MOUTH_POINTS
)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask_v2(shape, landmarks_aligned, mat, size):
    OVERLAY_POINTS = [
        LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54]
    ]

    pts = []
    # draw contours around cheek
    # right side
    w = landmarks_aligned[48][0] - landmarks_aligned[17][0]
    h = landmarks_aligned[48][1] - landmarks_aligned[17][1]

    x = landmarks_aligned[17][0]
    for i in range(1, 5):
        x = x + i * (w / 15)
        y = landmarks_aligned[17][1] + i * (h / 5)
        pts.append([x, y])

    w = landmarks_aligned[26][0] - landmarks_aligned[54][0]
    h = landmarks_aligned[54][1] - landmarks_aligned[26][1]

    x = landmarks_aligned[26][0]
    for i in range(1, 5):
        x = x - i * (w / 15)
        y = landmarks_aligned[26][1] + i * (h / 5)
        pts.append([x, y])

    for group in OVERLAY_POINTS:
        pts = np.concatenate([pts, landmarks_aligned[group]], 0)

    tmp = np.concatenate([np.transpose(pts), np.ones([1, pts.shape[0]])], 0)
    # Transform back to original location
    ary = np.expand_dims(np.array([0, 0, 1]), axis=0)
    mat = np.concatenate([mat * size, ary], 0)
    pts_org = np.dot(np.linalg.inv(mat), tmp)
    pts_org = pts_org[:2, :]
    pts_org = np.transpose(pts_org)
    im = np.zeros(shape, dtype=np.float64)

    draw_convex_hull(im, np.int32(pts_org), color=1)

    # im = np.array([im, im, im], dtype=np.uint8).transpose((1, 2, 0))

    return im


def bur_size(landmarks):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks[LEFT_EYE_POINTS], axis=0)
        - np.mean(landmarks[RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    return blur_amount


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0)
        - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0)
    )
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (
        im2.astype(np.float64)
        * im1_blur.astype(np.float64)
        / im2_blur.astype(np.float64)
    )


#  RoIs are chosen as the rectangle areas that contains both the face and surrounding areas.
#  Resized to 224x224 after cutting to feed into the network
def cut_head(imgs, point, seed=None):
    h, w = imgs[0].shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    delta_x = (x2 - x1) / 8
    delta_y = (y2 - y1) / 5
    if seed is not None:
        np.random.seed(seed)
    delta_x = np.random.randint(delta_x)
    delta_y = np.random.randint(delta_y)
    x1_ = np.int(np.maximum(0, x1 - delta_x))
    x2_ = np.int(np.minimum(w - 1, x2 + delta_x))
    y1_ = np.int(np.maximum(0, y1 - delta_y))
    y2_ = np.int(np.minimum(h - 1, y2 + delta_y * 0.5))
    imgs_new = []
    for i, im in enumerate(imgs):
        im = im[y1_:y2_, x1_:x2_, :]
        imgs_new.append(im)
    locs = [x1_, y1_, x2_, y2_]
    return imgs_new, locs


# Returns a list of bounding boxes of faces in the image
def get_face_loc(im, face_detector, scale=0):
    """ get face locations, color order of images is rgb """
    faces = face_detector(np.uint8(im), scale)
    face_list = []
    if faces is not None or len(faces) > 0:
        for i, d in enumerate(faces):
            try:
                face_list.append([d.left(), d.top(), d.right(), d.bottom()])
            except:
                face_list.append(
                    [d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()]
                )
    return face_list


# Returns a mask for all faces of given shape, face list
def get_all_face_mask(shape, face_list):

    # draws a convex hull on the face with given overlay points
    def get_face_mask(shape, landmarks):
        OVERLAY_POINTS = [
            LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54]
        ]
        im = np.zeros(shape, dtype=np.float64)

        for group in OVERLAY_POINTS:
            draw_convex_hull(im, landmarks[group], color=1)

        im = np.array([im, im, im]).transpose((1, 2, 0))
        return im

    mask = np.zeros(shape)
    for _, points in face_list:
        mask += np.int32(get_face_mask(shape[:2], points))

    mask = np.uint8(mask > 0)
    return mask


# Returns a list of trans-matrix, (x,y) landmark points for each face
def align(im, face_detector, lmark_predictor, scale=0):
    # This version we handle all faces in view
    # channel order rgb

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    def shape_to_np(shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

    im = np.uint8(im)
    faces = face_detector(im, scale)
    face_list = []
    if faces:
        for pred in faces:
            # get x,y coordinates
            points = shape_to_np(lmark_predictor(im, pred))
            # find transformation matrix to the mean landmarks
            trans_matrix = umeyama(points[17:], landmarks_2D, True)[0:2]
            face_list.append([trans_matrix, points])
    return face_list


def get_2d_aligned_face(image, mat, size, padding=[0, 0]):
    mat = mat * size
    mat[0, 2] += padding[0]
    mat[1, 2] += padding[1]
    return cv2.warpAffine(image, mat, (size + 2 * padding[0], size + 2 * padding[1]))


def get_aligned_face_and_landmarks(
    im, face_cache, aligned_face_size=256, padding=(0, 0)
):
    """
    get all aligned faces and landmarks of all images
    :param imgs: origin images
    :param fa: face_alignment package
    :return:
    """

    aligned_cur_shapes = []
    aligned_cur_im = []
    for mat, points in face_cache:
        # Get transform matrix
        aligned_face = get_2d_aligned_face(im, mat, aligned_face_size, padding)
        # Mapping landmarks to aligned face
        pred_ = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
        pred_ = np.transpose(pred_)
        mat = mat * aligned_face_size
        mat[0, 2] += padding[0]
        mat[1, 2] += padding[1]
        aligned_pred = np.dot(mat, pred_)
        aligned_pred = np.transpose(aligned_pred[:2, :])
        aligned_cur_shapes.append(aligned_pred)
        aligned_cur_im.append(aligned_face)

    return aligned_cur_im, aligned_cur_shapes


def get_aligned_mask(size):
    a = align(cimg[:, :, (2, 1, 0)], front_face_detector, lmark_predictor)
    b, c = get_aligned_face_and_landmarks(cimg, a, size)
    e = get_face_mask_v2(cimg.shape[:2], c[0], a[0][0], size)
    f = cv2.bitwise_and(cimg, cimg, mask=e.astype(np.uint8))
    return f


def apply_mask_to_img(img, mask):
    g = cv2.subtract(img, mask)
    h = cv2.add(g, mask)
    return h


def compose_mask(mask):
    og_dim = mask.shape
    aug = A.Compose(
        [
            A.Downscale(
                scale_min=0.15,
                scale_max=0.5,
                interpolation=cv2.INTER_AREA,
                always_apply=True,
                p=1,
            ),
            A.RandomGamma(gamma_limit=(80, 90), p=1),
            A.RandomBrightnessContrast(
                brightness_limit=0.1, contrast_limit=0.1, always_apply=True, p=1
            ),
            A.HueSaturationValue(
                hue_shift_limit=2,
                sat_shift_limit=10,
                val_shift_limit=2,
                always_apply=True,
                p=1,
            ),
            A.OpticalDistortion(
                distort_limit=(0, 0.6),
                shift_limit=(0, 0.6),
                interpolation=cv2.INTER_AREA,
                always_apply=True,
                p=1,
            ),
            A.RGBShift(r_shift_limit=2, g_shift_limit=2, b_shift_limit=2, p=0.1),
            A.ElasticTransform(
                alpha=2, sigma=10, alpha_affine=10, interpolation=cv2.INTER_AREA, p=0.01
            ),
            A.OneOf(
                [
                    A.Blur(blur_limit=(7, 15), always_apply=True, p=1),
                    A.MotionBlur(blur_limit=(7, 15), always_apply=True, p=1),
                    A.MedianBlur(blur_limit=(7, 15), always_apply=True, p=1),
                    A.GaussianBlur(blur_limit=(7, 15), always_apply=True, p=1),
                ],
                p=1,
            ),
        ],
        p=1,
    )
    f = aug(image=mask)["image"]
    f = cv2.resize(f, (og_dim[1], og_dim[0]), interpolation=cv2.INTER_AREA)
    return f


front_face_detector = dlib.get_frontal_face_detector()
lmark_predictor = dlib.shape_predictor(
    "/home/teh_devs/deepfake/CVPRW2019_Face_Artifacts/dlib_model/shape_predictor_68_face_landmarks.dat"
)

reals, fakes = get_split_df()
random.seed(42)
# a = random.choice(reals[0])
dest = "/home/teh_devs/deepfake/dataset/finale/"
out_of_bounds = set()
for a in tqdm(reals[3]):
    a_folder = "/home/teh_devs/deepfake/dataset/revamp/" + a.split(".")[0]
    imgs = glob(f"{a_folder}/*.png")
    groups = set()
    for img in imgs:
        groups.add(img.split("_")[-2])
    for group in groups:
        imgs_this_group = sorted(glob(f"{a_folder}/*_{group}_*.png"))
        gsize = len(imgs_this_group)
        if gsize >= 10:
            save_imgs = imgs_this_group[gsize // 2 - 2: gsize // 2 + 2]
        elif gsize >= 5:
            save_imgs = imgs_this_group[gsize // 2 - 1: gsize // 2 + 1]
        else:
            save_imgs = imgs_this_group
        for si in save_imgs:
            cimg = cv2.imread(si)
            og = None
            if cimg.shape[0] < 80 or cimg.shape[1] < 80:
                og = cimg.shape
                cimg = image_resize(cimg, height=120)
            try:
                mask = get_aligned_mask(cimg.shape[0])
                f = compose_mask(mask)
                final = apply_mask_to_img(cimg, f)
                if og:
                    final = cv2.resize(final, (og[1], og[0]))
                og_name = si.split('/')[-1]
                filename = os.path.join(dest, og_name)
                cv2.imwrite(filename, final)
            except IndexError:
                out_of_bounds.add(a)
            except Exception as e:
                print("Couldn't process file ", si)
                print(e)

import torch
torch.save(out_of_bounds, "out_of_bounds4.pkl")
