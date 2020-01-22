import torch
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from glob import glob
import pandas as pd
import os
import copy
import h5py
from hp import hp


def load_data_imagefolder(train_data_transform, test_data_transform):
    print("Loading data")
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    img_dataset = datasets.ImageFolder(hp.data_dir, train_data_transform)
    print(img_dataset.class_to_idx)
    dataset_size = len(img_dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(
        indices,
        random_state=hp.seed,
        test_size=hp.test_split_percent,
        stratify=img_dataset.targets,
    )

    test_img_dataset = copy.deepcopy(img_dataset)
    test_img_dataset.transform = test_data_transform

    train_dataset = Subset(img_dataset, train_indices)
    test_dataset = Subset(test_img_dataset, test_indices)

    if hp.balanced_sampling:
        weights = make_weights_for_balanced_classes(
            img_dataset.targets, train_indices, len(img_dataset.classes)
        )
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        num_workers=hp.data_num_workers,
        batch_size=hp.train_batch_size,
        pin_memory=hp.use_pinned_memory_train,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=hp.data_num_workers,
        batch_size=hp.test_batch_size,
        pin_memory=hp.use_pinned_memory_test,
    )

    dataset_dict = {"train": train_dataset, "test": test_dataset}
    dataloaders = {"train": train_loader, "test": test_loader}
    print("Done loading data")
    return dataset_dict, dataloaders


def load_hdf_data(key):
    print("Loading data")
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)

    flo_dataset = HDFDataset(hp.data_dir, key)
    print(flo_dataset.class_to_idx)
    dataset_size = len(flo_dataset)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(
        indices,
        random_state=hp.seed,
        test_size=hp.test_split_percent,
        stratify=flo_dataset.targets,
    )

    train_dataset = Subset(flo_dataset, train_indices)
    test_dataset = Subset(flo_dataset, test_indices)

    if hp.balanced_sampling:
        weights = make_weights_for_balanced_classes(
            flo_dataset.targets, train_indices, len(flo_dataset.classes)
        )
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        num_workers=hp.data_num_workers,
        batch_size=hp.train_batch_size,
        pin_memory=hp.use_pinned_memory_train,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=hp.data_num_workers,
        batch_size=hp.test_batch_size,
        pin_memory=hp.use_pinned_memory_test,
    )

    dataset_dict = {"train": train_dataset, "test": test_dataset}
    dataloaders = {"train": train_loader, "test": test_loader}
    print("Done loading data")
    return dataset_dict, dataloaders


class FloDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super(FloDataset, self).__init__()
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            raise RuntimeError()

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        file_list = []
        dir = os.path.expanduser(dir)
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if path.split(".")[-1] == "flo":
                        item = (path, class_to_idx[target])
                        file_list.append(item)

        return file_list

    def __getitem__(self, index):
        path, target = self.samples[index]
        with open(path, "rb") as f:
            magic, = np.fromfile(f, np.float32, count=1)
            if 202021.25 != magic:
                raise RuntimeError(f"Magic number incorrect. Invalid .flo file {path}")
            else:
                w, h = np.fromfile(f, np.int32, count=2)
                # print(f"Reading {w} x {h} flo file")
                data = np.fromfile(f, np.float32, count=2 * w * h)
                data2D = np.resize(data, (h, w, 2))

                # print("Normalized per channel!!!!")
                channel_mean = data2D.reshape(-1, 2).mean(0)  # 1x2
                channel_std = data2D.reshape(-1, 2).std(0)  # 1x2
                data2D -= channel_mean
                data2D /= channel_std
                sample = torch.from_numpy(data2D).permute(2, 0, 1)  # Channels first

        return sample, target

    def __len__(self):
        return len(self.samples)


class HDFDataset(torch.utils.data.Dataset):
    """
    HDF File of following format
    num_fake -> single uint32 number of fakes
    num_real -> single uint32 number of real
    total = num_fake + num_real
    key -> total x w x h x dim data, np.float32 (fakes first then real)
    mean -> (2,) np.float64, mean per channel
    std -> (2,) np.float64, std per channel
    """

    def __init__(self, root, key):
        super(HDFDataset, self).__init__()
        self.root = root

        self.key = key
        self.num_real_key = "num_real"
        self.num_fake_key = "num_fake"
        self.mean_key = "mean"
        self.std_key = "std"

        with h5py.File(self.root, "r") as hdfile:
            self.num_fake = hdfile[self.num_fake_key][()][0]
            self.num_real = hdfile[self.num_real_key][()][0]
            self.mean = hdfile[self.mean_key][()]
            self.std = hdfile[self.std_key][()]

        self.classes = ["fake", "real"]
        self.class_to_idx = {"fake": 0, "real": 1}
        self.targets = [0 for _ in range(self.num_fake)] + [
            1 for _ in range(self.num_real)
        ]

    def __getitem__(self, index):
        if index >= self.num_fake:
            target = 1
        else:
            target = 0

        with h5py.File(self.root, "r") as hdfile:
            sample = hdfile[self.key][index]

        sample -= self.mean
        sample /= self.std
        if self.key == "flow":
            sample = torch.from_numpy(sample).permute(2, 0, 1)  # Channels first
        else:
            sample = torch.from_numpy(sample).unsqueeze(0)

        return sample, target

    def __len__(self):
        return self.num_fake + self.num_real


def make_weights_for_balanced_classes(targets, indices, nclasses):
    count = [0] * nclasses
    for index in indices:
        count[targets[index]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(indices)
    for idx, val in enumerate(indices):
        weight[idx] = weight_per_class[targets[val]]
    return weight


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
