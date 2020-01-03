from albumentations import (
    Compose,
    OneOf,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    Normalize,
    HorizontalFlip,
    Resize,
    Rotate,
    JpegCompression,
    ChannelShuffle,
    InvertImg,
    RandomBrightnessContrast,
    RGBShift,
    RandomGamma,
    HueSaturationValue,
    Blur,
    CLAHE,
    GridDistortion,
    MedianBlur,
    OpticalDistortion,
    ShiftScaleRotate,
    IAAEmboss,
    IAAPiecewiseAffine,
    IAASharpen,
    Flip,
    MotionBlur,
    RandomRotate90,
    Transpose,
)
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np


def get_image_transform_no_crop_scale(image_size, mean, std):
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # cust_transform = [
    #     # transforms.Lambda(lambda x: np.array(x)),
    #     Resize(*image_size, interpolation=cv2.INTER_AREA),
    #     Rotate(limit=8, p=0.4),
    #     HorizontalFlip(p=0.4),
    #     JpegCompression(quality_lower=25, quality_upper=75, p=0.4),
    #     RandomBrightnessContrast(brightness_limit=(0.9, 1.2), contrast_limit=(0.9, 1.2), p=0.4),
    #     RGBShift(p=0.4),
    #     RandomGamma(p=0.4),
    #     HueSaturationValue(p=0.4),
    #     ChannelShuffle(p=0.2),
    #     OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.4),
    #     InvertImg(p=0.1),
    #     Normalize(mean=mean, std=std),
    #     ToTensorV2(),
    # ]
    # t = Compose(cust_transform)
    # return lambda img: t(image=np.array(img))


def get_test_transform(image_size, mean, std):
    return transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize(mean, std)])


def strong_aug(p=0.5):
    return Compose(
        [
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
            OneOf([MotionBlur(p=0.2), MedianBlur(blur_limit=3, p=0.1), Blur(blur_limit=3, p=0.1)], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([OpticalDistortion(p=0.3), GridDistortion(p=0.1), IAAPiecewiseAffine(p=0.3)], p=0.2),
            OneOf([CLAHE(clip_limit=2), IAASharpen(), IAAEmboss(), RandomBrightnessContrast()], p=0.3),
            HueSaturationValue(p=0.3),
        ],
        p=p,
    )

