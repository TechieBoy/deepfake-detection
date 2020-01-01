import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import copy
from tqdm import tqdm
from load_data import load_data_imagefolder
from meso import get_model
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from transforms import get_image_transform_no_crop_scale


device = torch.device("cuda:0")

# sorted
classes = ["fake", "real"]


def train_model(
    model, datasets, dataloaders, criterion, optimizer, scheduler, num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        probabilites = torch.zeros(0, dtype=torch.float32, device="cpu")
        predictions = torch.zeros(0, dtype=torch.long, device="cpu")
        true_val = torch.zeros(0, dtype=torch.long, device="cpu")

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    else:
                        # Softmax and calculate metrics
                        softmaxed = nn.functional.softmax(outputs, dim=1)
                        # Probability of class 1
                        probs = softmaxed[:, 1]
                        probabilites = torch.cat([probabilites, probs.view(-1).cpu()])
                        predictions = torch.cat([predictions, preds.view(-1).cpu()])
                        true_val = torch.cat([true_val, labels.view(-1).cpu()])

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            dataset_size = len(datasets[phase])
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "test":
                scheduler.step(epoch_loss)

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"meso_epoch_{epoch}.pt")

        true_val = true_val.numpy()
        predictions = predictions.numpy()
        probabilites = probabilites.numpy()
        print(classification_report(true_val, predictions, target_names=classes))
        print("Confusion Matrix")
        print(confusion_matrix(true_val, predictions))
        print("RoC AUC:")
        print(roc_auc_score(true_val, probabilites))
        print()
        probabilites = np.around(probabilites, deciamls=1)
        hist, _ = np.histogram(probabilites, bins=3)
        print("# Predictions magnitude below 0.33", hist[0])
        print("# Predictions magnitude 0.33 to 0.66", hist[1])
        print("# Predictions magnitude above 0.66", hist[2])

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


def run():
    model = get_model(2)
    image_size = model.get_image_size()
    mean, std = model.get_mean_std()
    data_transform = get_image_transform_no_crop_scale(image_size, mean, std)

    datasets, dataloaders = load_data_imagefolder(
        data_dir="../dataset/new",
        data_transform=data_transform,
        num_workers=70,
        train_batch_size=500,
        test_batch_size=500,
        seed=420,
        test_split_size=0.25,
    )
    print("Classes array (check order)")
    print(classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = train_model(
        model, datasets, dataloaders, criterion, optimizer, scheduler, 50
    )

    torch.save(model.state_dict(), "meso.pt")


if __name__ == "__main__":
    run()
