import torch
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from glob import glob
import pandas as pd
import os
import copy


def load_data_imagefolder(
    data_dir,
    data_transform,
    num_workers,
    train_batch_size,
    test_batch_size,
    seed,
    test_split_size,
):
    print("Loading data")
    np.random.seed(seed)
    torch.manual_seed(seed)
    img_dataset = datasets.ImageFolder(data_dir, data_transform)
    print(img_dataset.class_to_idx)
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(
        indices,
        random_state=seed,
        test_size=test_split_size,
        stratify=img_dataset.targets,
    )

    test_img_dataset = copy.deepcopy(img_dataset)
    test_img_dataset.transform = None
    test_img_dataset.transforms = None

    train_dataset = Subset(img_dataset, train_indices)
    test_dataset = Subset(test_img_dataset, test_indices)

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

    weights = make_weights_for_balanced_classes(
        img_dataset.imgs, train_indices, len(img_dataset.classes)
    )
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        num_workers=num_workers,
        batch_size=train_batch_size,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=num_workers,
        batch_size=test_batch_size,
        pin_memory=True,
    )

    dataset_dict = {"train": train_dataset, "test": test_dataset}
    dataloaders = {"train": train_loader, "test": test_loader}
    print("Done loading data")
    return dataset_dict, dataloaders


def combine_metadata():
    folders = glob("/home/teh_devs/deepfake/raw/*")
    dfs = []
    for f in folders:
        if os.path.isdir(f):
            g = pd.read_json(os.path.join(f, "metadata.json")).T.reset_index()
            g["folder"] = f
            g = g[g["split"] != "test"]
            g = g.drop(columns=["split"])
            dfs.append(g)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df.to_csv("combined_metadata.csv", index=False)


if __name__ == "__main__":
    combine_metadata()
