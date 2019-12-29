from xception import xception
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
import time
import copy
from tqdm import tqdm
from sklearn.model_selection import train_test_split


data_transform = transforms.Compose(
    [transforms.Resize((299, 299)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
)

data_dir = "../dataset/train/"

num_workers = 35
train_batch_size = 488
test_batch_size = 488
img_dataset = datasets.ImageFolder(data_dir, data_transform)

dataset_size = len(img_dataset)
indices = list(range(dataset_size))
train_indices, test_indices = train_test_split(indices, random_state=69, test_size=0.1, stratify=img_dataset.targets)

train_dataset = Subset(img_dataset, train_indices)
test_dataset = Subset(img_dataset, test_indices)


def make_weights_for_balanced_classes(images, indices, nclasses):
    count = [0] * nclasses
    for index in indices:
        count[images[index][1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(indices)
    for idx, val in enumerate(indices):
        weight[idx] = weight_per_class[images[val][1]]
    return weight


weights = make_weights_for_balanced_classes(img_dataset.imgs, train_indices, len(img_dataset.classes))
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(train_dataset, sampler=sampler, num_workers=num_workers, batch_size=train_batch_size, pin_memory=True)

test_loader = DataLoader(test_dataset, shuffle=False, num_workers=num_workers, batch_size=test_batch_size, pin_memory=True)


dataloaders = {"train": train_loader, "test": test_loader}
dataset_sizes = {"train": len(train_dataset), "test": len(test_dataset)}


def train_model(model, device, criterion, optimizer, scheduler, num_epochs=25):
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
                torch.save(best_model_wts, f"xception_epoch_{epoch}.pt")

        print()

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run():
    criterion = nn.CrossEntropyLoss()
    model = xception()

    device = torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        cudnn.benchmark = True
        cudnn.deterministic = True

    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        torch.distributed.init_process_group(backend="nccl")
        model = nn.parallel.DistributedDataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    exp_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, verbose=True)

    model_v1 = train_model(model, device, criterion, optimizer, exp_lr_scheduler, 25)

    torch.save(model_v1.state_dict(), "xception.pt")


if __name__ == "__main__":
    print(img_dataset.class_to_idx)
    run()
    # To load distributed dataparallel back to normal
    # state_dict = torch.load('/kaggle/input/resnet101-meso-trained-26dec-deepfake/xception_epoch_2.pt', map_location=torch.device('cpu') )

    # # create new OrderedDict that does not contain `module.`
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:] # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)
