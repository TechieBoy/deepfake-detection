from albumentations import (
    Compose,
    OneOf,
    CenterCrop,
    PadIfNeeded,
    Downscale,
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
from hp import hp


def get_image_transform_no_crop_scale(image_size, mean, std):
    if hp.using_augments:
        return transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        cust_transform = [
            # transforms.Lambda(lambda x: np.array(x)),
            Resize(*image_size, interpolation=cv2.INTER_AREA),
            Rotate(limit=8, p=0.4),
            HorizontalFlip(p=0.4),
            JpegCompression(quality_lower=25, quality_upper=75, p=0.4),
            RandomBrightnessContrast(
                brightness_limit=(0.9, 1.2), contrast_limit=(0.9, 1.2), p=0.4
            ),
            RGBShift(p=0.4),
            RandomGamma(p=0.4),
            HueSaturationValue(p=0.4),
            ChannelShuffle(p=0.2),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.4),
            InvertImg(p=0.1),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
        t = Compose(cust_transform)
        return lambda img: t(image=np.array(img))


def get_test_transform(image_size, mean, std):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def get_test_transform_albumentations(image_size, mean, std):
    return Compose(
        [
            Resize(*image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def train_albumentations(image_size, mean, std):
    return Compose(
        [
            OneOf(
                [
                    Compose([PadIfNeeded(50, 50, p=1), CenterCrop(30, 30, p=1)], p=0.5),
                    Downscale(p=0.5),
                ],
                p=1,
            ),
            Rotate(limit=8, p=0.4),
            HorizontalFlip(p=0.6),
            JpegCompression(quality_lower=25, quality_upper=65, p=0.4),
            RandomBrightnessContrast(
                brightness_limit=(0.9, 1.2), contrast_limit=(0.9, 1.2), p=0.4
            ),
            RGBShift(p=0.4),
            RandomGamma(p=0.4),
            HueSaturationValue(p=0.4),
            ChannelShuffle(p=0.1),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.4),
            InvertImg(p=0.1),
            Resize(*image_size, interpolation=cv2.INTER_AREA),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
