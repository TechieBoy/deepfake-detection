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
)
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import cv2
import numpy as np


def get_image_transform_no_crop_scale(image_size, mean, std):
    return Compose(
        [
            transforms.Lambda(lambda x: np.array(x)),
            Resize(*image_size, interpolation=cv2.INTER_AREA),
            Rotate(limit=10, p=0.4),
            HorizontalFlip(p=0.4),
            JpegCompression(quality_lower=35, quality_upper=100, p=0.4),
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
    )
