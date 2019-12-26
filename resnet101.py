import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import ImageFile
import cvlib as cv
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True

# GPU
device = torch.device("cuda:0")
cudnn.benchmark = True
cudnn.deterministic = True

data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_dir = "/home/teh_devs/deepfake/dataset"
data_folders = ["train", "test"]
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform)
    for x in data_folders
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=234, shuffle=True, num_workers=78, pin_memory=True
    )
    for x in data_folders
}

dataset_sizes = {x: len(image_datasets[x]) for x in data_folders}
class_names = image_datasets["train"].classes


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
                torch.save(best_model_wts, f"resnet101_epoch_{epoch}.pt")

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_model():
    model_v1 = models.resnet101(pretrained=False)
    for param in model_v1.parameters():
        param.requires_grad = True

    num_last_layer = model_v1.fc.in_features
    model_v1.fc = nn.Linear(num_last_layer, 2)

    model_v1 = model_v1.to(device)
    return model_v1

def run():
    criterion = nn.CrossEntropyLoss()
    model_v1 = get_model()
    optimizer = optim.SGD(model_v1.parameters(), lr=0.001, momentum=0.9)

    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, verbose=True
    )

    model_v1 = train_model(model_v1, criterion, optimizer, exp_lr_scheduler, 25)

    torch.save(model_v1.state_dict(), "resnet101.pt")

if __name__ == '__main__':
    print(image_datasets['train'].class_to_idx)