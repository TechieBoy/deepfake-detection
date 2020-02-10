import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import os
import shutil
from datetime import datetime
from tqdm import tqdm
from load_data import load_data_imagefolder, load_hdf_data, load_split_data, load_fwa_data, load_split_data_all
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from transforms import (
    get_test_transform,
    train_albumentations,
    get_test_transform_albumentations,
)
import math
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from hp import hp

# Disable if input sizes vary a lot
torch.backends.cudnn.benchmark = True

if not os.path.exists(hp.save_folder):
    os.mkdir(hp.save_folder)


now = datetime.now()
writer = None
if not (hp.quick_run or hp.pre_run):
    writer = SummaryWriter(os.path.join("runs", hp.model_name, now.strftime("%d%b%I:%M%p")))

device = hp.device

# sorted
classes = ["fake", "real"]


def load_multi_gpu(model):
    model.to(device)
    if False or torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # torch.distributed.init_process_group(backend="nccl")
        # model = nn.parallel.DistributedDataParallel(model)
        model = nn.DataParallel(model, hp.device_ids)
    else:
        print(r"Multiple GPU's not available (╯°□°）╯︵ ┻━┻")
    return model


def train_model(
    model, datasets, dataloaders, criterion, optimizer, scheduler, num_epochs, save_loc
):
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        if hp.use_cos_anneal_restart:
            optimizer = optim.SGD(model.parameters(), **hp.sgd_params)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=len(dataloaders["train"]), **hp.cos_anneal_sched_params
            )
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)
        print("LR", optimizer.param_groups[0]["lr"])

        probabilites = torch.zeros(0, dtype=torch.float32, device="cpu")
        predictions = torch.zeros(0, dtype=torch.long, device="cpu")
        true_val = torch.zeros(0, dtype=torch.long, device="cpu")

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            n_iter_this_phase = 0
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                if hp.quick_run:
                    break
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], ncols=0, mininterval=1):
                if hp.quick_run and n_iter_this_phase >= 150:
                    break
                n_iter_this_phase += 1

                non_blocking_transfer = (
                    True if phase == "train" and hp.use_pinned_memory_train else False
                )
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
                        if hp.use_one_cycle_lr or hp.use_cos_anneal_restart:
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
            if not hp.quick_run:
                writer.add_scalar(f"Loss/{phase}", epoch_loss, epoch)
                writer.add_scalar(f"Accuracy/{phase}", epoch_acc, epoch)

            print("{} Loss: {:.8f} Acc: {:.8f}".format(phase, epoch_loss, epoch_acc))

            if phase == "test":
                if hp.use_step_lr:
                    scheduler.step()
                elif hp.use_plateau_lr:
                    # Passing in test loss!
                    scheduler.step(epoch_loss)

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
            if not hp.quick_run:
                file_name = f"{hp.model_name}_{epoch+1}.pt"
                torch.save(model.state_dict(), os.path.join(save_loc, file_name))
        if not hp.quick_run:
            true_val = true_val.numpy()
            predictions = predictions.numpy()
            probabilites = probabilites.numpy()
            print(classification_report(true_val, predictions, target_names=classes))
            
            writer.add_pr_curve("pr_curve", true_val, probabilites, epoch)
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
    return model


def load_data_for_model(model):
    image_size = model.get_image_size()
    mean, std = model.get_mean_std()
    if hp.using_hdf:
        return load_hdf_data(hp.hdf_key)
    elif hp.using_split:
        train_data_transform = train_albumentations(image_size, mean, std)
        test_data_transform = get_test_transform_albumentations(image_size, mean, std)
        return load_split_data_all(train_data_transform, test_data_transform)
    elif hp.using_fwa:
        train_data_transform = train_albumentations(image_size, mean, std)
        test_data_transform = get_test_transform_albumentations(image_size, mean, std)
        return load_fwa_data(train_data_transform, test_data_transform)
    else:
        test_transform = get_test_transform(image_size, mean, std)
        data_transform = get_test_transform(image_size, mean, std)
        return load_data_imagefolder(
            train_data_transform=data_transform, test_data_transform=test_transform
        )


def pre_run():
    print("-------------------------------------")
    print(f"Now Pre-running model: {hp.model_name}")
    print("-------------------------------------")
    model = hp.model
    datasets, dataloaders = load_data_for_model(model)
    model = model.to(device)
    model = load_multi_gpu(model)
    weights = None
    if hp.use_class_weights:
        weights = torch.FloatTensor(hp.class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Find LR first
    logs, losses = find_lr(model, criterion, dataloaders["train"])
    torch.save(logs, "pre_run_logs.pt")
    torch.save(losses, "pre_run_losses.pt")
    # plt.plot(logs, losses)


def run():
    print("-------------------------------------")
    print(f"Now running model: {hp.model_name}")
    print(now.strftime("%A, %d %B at %I:%M %p"))
    print("-------------------------------------")

    save_loc = hp.model_name + now.strftime("%d%b%I:%M%p")
    save_loc = os.path.join(hp.save_folder, save_loc)
    if not hp.quick_run:
        if not os.path.isdir(save_loc):
            os.mkdir(save_loc)
        shutil.copy("hp.py", save_loc)
    model = hp.model
    datasets, dataloaders = load_data_for_model(model)

    print("Classes array (check order)")
    print(classes)

    model = model.to(device)
    model = load_multi_gpu(model)

    weights = None
    if hp.use_class_weights:
        weights = torch.FloatTensor(hp.class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    if hp.use_adamW:
        optimizer = optim.AdamW(model.parameters(), **hp.adamW_params)
    elif hp.use_sgd:
        optimizer = optim.SGD(model.parameters(), **hp.sgd_params)
    else:
        optimizer = None

    if hp.use_step_lr:
        scheduler = optim.lr_scheduler.StepLR(optimizer, **hp.step_sched_params)
    elif hp.use_plateau_lr:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **hp.plateau_lr_sched_params
        )
    elif hp.use_one_cycle_lr:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, steps_per_epoch=len(dataloaders["train"]), **hp.oc_sched_params
        )
    else:
        scheduler = None

    model = train_model(
        model,
        datasets,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        hp.num_epochs,
        save_loc,
    )


def find_lr(net, criterion, trn_loader, init_value=1e-8, final_value=10.0, beta=0.98):
    """
    https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    https://sgugger.github.io/the-1cycle-policy.html
    logs,losses = find_lr()
    plt.plot(logs,losses)
    For a OneCycleLR, The maximum should be the value picked with the Learning Rate Finder, and the lower one can be ten times lower.
    """
    if hp.use_adamW:
        optimizer = optim.AdamW(net.parameters(), **hp.adamW_params)
    elif hp.use_sgd:
        optimizer = optim.SGD(net.parameters(), **hp.sgd_params)
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
        inputs = inputs.to(device, non_blocking=hp.use_pinned_memory_train)
        labels = labels.to(device, non_blocking=hp.use_pinned_memory_train)

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
    if hp.pre_run:
        pre_run()
    else:
        run()
    writer.close()
