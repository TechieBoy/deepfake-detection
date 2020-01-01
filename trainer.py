import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import time
import copy
from tqdm import tqdm
from load_data import load_data_imagefolder
from meso import get_model, get_mean_std, get_image_size
from albumentations import (
    Compose,
    RandomCrop,
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


device = torch.device("cuda:0")

mean, std = get_mean_std()
image_size = get_image_size()

data_transform = Compose(
    [
        transforms.Lambda(lambda x: np.array(x)),
        Resize(*image_size, interpolation=cv2.INTER_AREA),
        Rotate(limit=10, p=0.4),
        HorizontalFlip(p=0.4),
        JpegCompression(quality_lower=35, quality_upper=100, p=0.4),
        RandomBrightnessContrast(brightness_limit=(0.9, 1.2), contrast_limit=(0.9, 1.2), p=0.4),
        RGBShift(p=0.4),
        RandomGamma(p=0.4),
        HueSaturationValue(p=0.4),
        ChannelShuffle(p=0.2),
        InvertImg(p=0.1),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
)


dataloaders, dataset_sizes = load_data_imagefolder(
    data_dir="../dataset/new",
    data_transform=data_transform,
    num_workers=70,
    train_batch_size=500,
    test_batch_size=500,
    seed=420,
    test_split_size=0.25,
)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "test":
                scheduler.step(epoch_loss)

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"meso_epoch_{epoch}.pt")

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run():
    model = get_model(2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = train_model(model, criterion, optimizer, scheduler, 50)

    torch.save(model.state_dict(), "meso.pt")


if __name__ == "__main__":
    print(datasets["train"].class_to_idx)

