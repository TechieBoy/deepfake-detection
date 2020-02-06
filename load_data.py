import torch
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader
from glob import glob
import pandas as pd
import os
import random
import math
import copy
import h5py
from hp import hp
import cv2


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


def load_fwa_data(train_data_transform, test_data_transform):
    print("Loading data")
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    random.seed(hp.seed)

    train_dataset = FWADataset(
        hp.real_folder_loc,
        hp.fake_loc,
        "train",
        hp.seed,
        hp.test_split_percent,
        train_data_transform,
    )
    test_dataset = FWADataset(
        hp.real_folder_loc,
        hp.fake_loc,
        "test",
        hp.seed,
        hp.test_split_percent,
        test_data_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=hp.data_num_workers,
        batch_size=hp.train_batch_size,
        pin_memory=hp.use_pinned_memory_train,
    )

    test_loader = DataLoader(
        test_dataset,
        num_workers=hp.data_num_workers,
        batch_size=hp.test_batch_size,
        pin_memory=hp.use_pinned_memory_test,
    )

    dataset_dict = {"train": train_dataset, "test": test_dataset}
    dataloaders = {"train": train_loader, "test": test_loader}
    print("Done loading data")
    return dataset_dict, dataloaders


def get_array_in_batch(arr, shuffle=False, seed=50, per=5000):
    if shuffle:
        random.seed(seed)
        random.shuffle(arr)
    div = math.ceil(len(arr) / per)
    batched = []
    for i in range(div):
        batched.append(arr[i * per : (i + 1) * per])
    return div, batched


def get_split_df(seed=50, per=5000):
    df = pd.read_csv(hp.split_csv)
    df = df[
        ((df.video_label == "FAKE") | (df.video_label == "REAL"))
        & (df.audio_label == "REAL")
    ]
    dff = df[(df.video_label == "FAKE")]
    df_reals = df[(df.video_label == "REAL")].filename.to_list()
    div, reals = get_array_in_batch(df_reals, shuffle=True, seed=seed, per=per)
    fakes = [[] for _ in range(div)]

    grouped = dff.groupby(dff.original)
    removed = []
    for i, rr in enumerate(reals):
        for r in rr:
            try:
                fakes[i].extend(grouped.get_group(r).filename.to_list())
            except KeyError:
                removed.append(r)
    return fakes, reals, removed


def load_split_data_all(train_data_transform, test_data_transform):
    fakes, reals, removed = get_split_df(hp.split_seed, hp.per)
    total_real = [subitem for item in reals for subitem in item]
    total_fake = [subitem for item in fakes for subitem in item]

    random.seed(hp.seed)
    random.shuffle(total_real)
    random.shuffle(total_fake)
    split_val_real = round(hp.test_split_percent * len(total_real))
    split_val_fake = round(hp.test_split_percent * len(total_fake))

    test_real, train_real = total_real[:split_val_real], total_real[split_val_real:]

    test_fake, train_fake = total_fake[:split_val_fake], total_fake[split_val_fake:]

    root = hp.data_dir
    print("Train")
    train_dataset = SplitDataset(root, train_fake, train_real, train_data_transform)
    print("Test")
    test_dataset = SplitDataset(root, test_fake, test_real, test_data_transform)

    train_loader = DataLoader(
        train_dataset,
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
    return dataset_dict, dataloaders


def load_split_data(train_data_transform, test_data_transform):
    root = hp.data_dir
    fakes, reals, removed = get_split_df(hp.split_seed, hp.per)
    total_train_reals = []
    total_train_fakes = []
    total_test_reals = []
    total_test_fakes = []

    def create_full_list(idxlist, total_real, total_fake):
        for ri, fi, fsi in idxlist:
            total_real.extend(reals[ri])
            div, batched_fake = get_array_in_batch(
                fakes[fi], hp.shuffle_fake, hp.shuffle_fake_seed, hp.per
            )
            total_fake.extend(batched_fake[fsi])

    create_full_list(hp.train_idx_list, total_train_reals, total_train_fakes)
    create_full_list(hp.test_idx_list, total_test_reals, total_test_fakes)
    print("Train")
    train_dataset = SplitDataset(
        root, total_train_fakes, total_train_reals, train_data_transform
    )
    print("Test")
    test_dataset = SplitDataset(
        root, total_test_fakes, total_test_reals, test_data_transform
    )

    train_loader = DataLoader(
        train_dataset,
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
    return dataset_dict, dataloaders


class FWADataset(torch.utils.data.Dataset):
    def __init__(self, real_folder_loc, fake_loc, mode, seed, percent, transform=None):
        super(FWADataset, self).__init__()
        self.transform = transform
        self.class_to_idx = {"fake": 0, "real": 1}
        self.mode = mode
        self.percent = percent
        self.seed = seed
        self.samples = self.make_dataset(real_folder_loc, fake_loc)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(sample[0])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, sample[1]

    def __len__(self):
        return len(self.samples)

    def make_dataset(self, real_folder_loc, fake_loc):
        sample_list = []
        fake_imgs = os.listdir(fake_loc)
        random.seed(self.seed)
        random.shuffle(fake_imgs)
        loc = round(self.percent * len(fake_imgs))
        if self.mode == "train":
            fake_imgs = fake_imgs[loc:]
        elif self.mode == "test":
            fake_imgs = fake_imgs[:loc]
        else:
            raise RuntimeError("Invalid Mode")
        print(f"{self.mode} {len(fake_imgs)}")
        for img in fake_imgs:
            fake_path = os.path.join(fake_loc, img)
            sample_list.append((fake_path, self.class_to_idx["fake"]))
            real_folder_name = img[:10]
            real_path = os.path.join(real_folder_loc, real_folder_name, img)
            sample_list.append((real_path, self.class_to_idx["real"]))
        return sample_list


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, root, fakes, reals, transform=None):
        super(SplitDataset, self).__init__()
        self.transform = transform
        self.class_to_idx = {"fake": 0, "real": 1}
        self.samples = self.make_dataset(root, reals, 1)
        print("real", len(self.samples))
        fake_samples = self.make_dataset(root, fakes, 0)
        print("fake", len(fake_samples))
        self.samples.extend(fake_samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(sample[0])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, sample[1]

    def __len__(self):
        return len(self.samples)

    def make_dataset(self, root, folders, target):
        sample_list = []
        for f in folders:
            f = f.split(".")[0]
            ff = os.path.join(root, f)
            if os.path.isdir(ff):
                images = os.listdir(ff)
                for img in images:
                    img_path = os.path.join(ff, img)
                    sample_list.append((img_path, target))
        return sample_list


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
    load_split_data_all(None, None)
