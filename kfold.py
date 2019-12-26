import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import time
import copy
from tqdm import tqdm
from PIL import ImageFile
from sklearn.model_selection import StratifiedKFold

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

data_dir = "/home/teh_devs/deepfake/dataset/full"
dataset = datasets.ImageFolder(data_dir)


def train_for_folds(kf, model, criterion, num_epochs_per_fold, num_workers, train_batch_size, test_batch_size=None):
    """Trains and saves n different models, where n is number of folds"""
    if test_batch_size is None:
        test_batch_size = train_batch_size

    og_model = copy.deepcopy(model)

    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, dataset.targets)):
        print(f"In fold {fold}")

        # Reset model
        model = og_model
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_loader = DataLoader(
            train_dataset, shuffle=False, num_workers=num_workers, batch_size=train_batch_size, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, shuffle=False, num_workers=num_workers, batch_size=test_batch_size, pin_memory=True
        )
        dataloaders = {"train": train_loader, "test": test_loader}
        dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}

        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        optimizer = optim.SGD(model_v1.parameters(), lr=0.001, momentum=0.9)

        # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.1, verbose=True)

        for epoch in range(num_epochs_per_fold):
            print(f"Epoch {epoch + 1}/{num_epochs_per_fold}")
            print("-" * 10)

            # Each epoch has a training and test phase
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
                    torch.save(best_model_wts, f"model_fold_{fold}_epoch_{epoch}.pt")

            print()

        time_elapsed = time.time() - since
        print("Fold complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val Acc: {:4f}".format(best_acc))

        # Save best model for this fold
        torch.save(best_model_wts, f"best_model_fold_{fold}.pt")


model_v1 = models.resnet101(pretrained=False)
for param in model_v1.parameters():
    param.requires_grad = True

num_last_layer = model_v1.fc.in_features
model_v1.fc = nn.Linear(num_last_layer, 2)

model_v1 = model_v1.to(device)
criterion = nn.CrossEntropyLoss()


skf = StratifiedKFold(5, True, 69)

torch.save(model_v1.state_dict(), "resnet101.pt")
