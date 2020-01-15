import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import copy
from tqdm import tqdm
from load_data import load_data_imagefolder, load_data_flofolder
from xception import get_model
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from transforms import get_image_transform_no_crop_scale, get_test_transform
import math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = torch.device("cuda:0")

# sorted
classes = ["fake", "real"]

USING_ALBUMENTATIONS = False
USE_PINNED_MEMORY = True


def load_multi_gpu(model):
    model.to(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # torch.distributed.init_process_group(backend="nccl")
        # model = nn.parallel.DistributedDataParallel(model)
        model = nn.DataParallel(model)
        return model
    else:
        raise AssertionError("Multiple GPU's not available")


def train_model(
    model, datasets, dataloaders, criterion, optimizer, scheduler, num_epochs=25
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    n_iter = 0

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

            for inputs, labels in tqdm(dataloaders[phase], ncols=0, mininterval=1):
                n_iter += 1
                non_blocking_transfer = True if phase == 'train' and USE_PINNED_MEMORY else False
                if phase == 'train' and USING_ALBUMENTATIONS:
                    inputs = inputs['image']
                inputs = inputs.to(device, non_blocking=non_blocking_transfer)
                labels = labels.to(device, non_blocking=non_blocking_transfer)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        # For OneCycleLR
                        scheduler.step()
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
            epoch_acc = float(running_corrects) / dataset_size
            writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
            writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # If using ReduceLRonPlateau or similar to be called every epoch
            # if phase == "test":
            #     scheduler.step(epoch_loss)

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f"meso_epoch_{epoch+1}.pt")

        true_val = true_val.numpy()
        predictions = predictions.numpy()
        probabilites = probabilites.numpy()
        print(classification_report(true_val, predictions, target_names=classes))
        writer.add_pr_curve('pr_curve', true_val, probabilites, epoch)
        print("Confusion Matrix")
        print(confusion_matrix(true_val, predictions))
        print()
        print("RoC AUC:")
        print(roc_auc_score(true_val, probabilites))
        print()
        probabilites = np.around(probabilites, decimals=1)
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


def load_data_for_model(model):
    image_size = model.get_image_size()
    mean, std = model.get_mean_std()
    test_transform = get_test_transform(image_size, mean, std)
    data_transform = get_image_transform_no_crop_scale(image_size, mean, std)

    # return load_data_imagefolder(
    #     data_dir="/data/deepfake/",
    #     train_data_transform=data_transform,
    #     test_data_transform=test_transform,
    #     use_pinned_memory=USE_PINNED_MEMORY,  # Only for train, test always uses non pinned
    #     num_workers=30,
    #     train_batch_size=120,
    #     test_batch_size=120,
    #     seed=420,
    #     test_split_size=0.20,
    # )

    return load_data_flofolder(
        data_dir="/home/teh_devs/deepfake/dataset/of",
        use_pinned_memory=USE_PINNED_MEMORY,
        num_workers=70,
        train_batch_size=120,
        test_batch_size=120,
        seed=420,
        test_split_size=0.20
    )


def pre_run():
    model = get_model(2, 2)
    datasets, dataloaders = load_data_for_model(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Find LR first
    logs, losses = find_lr(model, criterion, dataloaders["train"])
    torch.save(logs, "pre_run_logs.pt")
    torch.save(losses, "pre_run_losses.pt")
    # plt.plot(logs, losses)


def run():
    model = get_model(2)
    datasets, dataloaders = load_data_for_model(model)

    print("Classes array (check order)")
    print(classes)

    model = model.to(device)
    model = load_multi_gpu(model)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=0.1, betas=(0.96, 0.99), weight_decay=0.05
    )

    num_epochs = 72
    # Have to call step every batch!!
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        div_factor=15.0,  # initial_lr = max_lr/div_factor
        final_div_factor=10000.0,  # min_lr = initial_lr/final_div_factor
        epochs=num_epochs,
        steps_per_epoch=len(dataloaders["train"]),
        pct_start=0.35,  # percentage of time going up/down
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=-1,  # Change this if resuming (Pass in total number of batches done, not epochs!!)
    )
    model = train_model(
        model, datasets, dataloaders, criterion, optimizer, scheduler, num_epochs
    )

    torch.save(model.state_dict(), "meso.pt")


def find_lr(net, criterion, trn_loader, init_value=1e-8, final_value=10.0, beta=0.98):
    """
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    https://sgugger.github.io/the-1cycle-policy.html
    logs,losses = find_lr()
    plt.plot(logs,losses)
    For a OneCycleLR, The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower.
    """
    optimizer = optim.SGD(net.parameters(), lr=1e-1)
    num = len(trn_loader) - 1
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0
    losses = []
    log_lrs = []
    for inputs, labels in tqdm(trn_loader, ncols=0, mininterval=1):
        batch_num += 1
        if USING_ALBUMENTATIONS:
            inputs = inputs["image"]
        inputs = inputs.to(device, non_blocking=USE_PINNED_MEMORY)
        labels = labels.to(device, non_blocking=USE_PINNED_MEMORY)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses

        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # Do the SGD step
        loss.backward()
        optimizer.step()

        # Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]["lr"] = lr
    return log_lrs[10:-5], losses[10:-5]


if __name__ == "__main__":
    pre_run()
    writer.close()
