from albumentations import (
    Compose,
    OneOf,
    CenterCrop,
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
    MultiplicativeNoise,
)
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np
from hp import hp


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
            Rotate(limit=6, p=0.5),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.2), contrast_limit=(-0.2, 0.2), p=0.3
            ),
            RGBShift(5, 5, 5, p=0.3),
            HueSaturationValue(1, 10, 5, p=0.2),
            GaussNoise(10, p=0.25),
            MultiplicativeNoise((0.85, 1.05), per_channel=True, p=0.25),
            ChannelShuffle(p=0.05),
            Resize(*image_size, interpolation=cv2.INTER_AREA),
            Compose(
                [
                    CenterCrop(width=200, height=200, p=1),
                    Resize(*image_size, interpolation=cv2.INTER_AREA),
                ],
                p=0.4,
            ),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
