""""
Here we implement a class for loading data.
"""

import pickle
import torch
import numpy as np
import sys
import torchvision.utils as vutils
from sklearn.model_selection import StratifiedShuffleSplit
import spectral
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

sys.path.append('/home/patrick/repositories/hyperspec')


# np.random.seed(0)


def load_dataset_train(filename):
    return filename


def normalize(data, max_value=None):
    # data -= np.min(data)
    if max_value is None:
        max_value = np.max(data)
    data /= max_value
    return data


class Hdr_dataset(Dataset):
    def __init__(self, train=True, load_to_mem=False):
        path = "/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/dataset"
        file_name = "dataset_small_10prozent.p"
        train_rowcol, train_lu_table, \
        test_rowcol, test_lu_table = self.load_leaf(os.path.join(path, file_name))

        self.lu_table = train_lu_table if train else test_lu_table
        self.rowcol = train_rowcol if train else test_rowcol  # [:1000]
        self.data = None
        self.labels = None
        self.n_bands = 442

        if load_to_mem:
            data_path = os.path.join(path, "info_gan/dataset_small_10prozent_{}.p".format("train" if train else "test"))
            if os.path.isfile(data_path):
                data = pickle.load(open(data_path, "rb"))
                self.data = data["x"]
                self.labels = data["y"]
            else:
                data = self.load_data_to_array()
                pickle.dump(data, open(data_path, "wb"))
                self.data = data["x"]
                self.labels = data["y"]
        # self.load_leaf()
        # compute which labels should be user for semi-supervised learning - by a mask
        """if sup_ratio < 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sup_ratio, random_state=1234)
            _, y_sup_mask_in = next(sss.split(self.x_train, self.y_train))
            y_sup_mast_indices = np.array(range(len(self.y_train)))
            self.y_sup_mask = np.in1d(y_sup_mast_indices, y_sup_mask_in)
        else:
            self.y_sup_mask = np.array([True for _ in range(len(self.y_train))])"""

    def __len__(self):
        return len(self.rowcol)

    def __getitem__(self, idx):
        if self.data is None or self.labels is None:
            return self.load_from_hdr(idx)
        else:
            return self.load_from_array(idx)

    def load_from_hdr(self, idx):
        (row, col, key_lu_table) = self.rowcol[idx]
        img = self.lu_table[key_lu_table]["hdr"]
        sample = img[row, col].squeeze()[range(0, self.n_bands, 3)]
        np.nan_to_num(sample, copy=False)
        sample[sample > 1] = 1
        sample[sample < 0] = 0
        return sample, (row, col, key_lu_table)

    def load_from_array(self, idx):
        return self.data[idx], self.labels[idx]

    def load_leaf(self, path):
        dataset_dict = pickle.load(
            open(path, "rb"))
        train_data = dataset_dict["train"]
        test_data = dataset_dict["test"]
        lu_table_train = dataset_dict["lookup_table_train"]
        lu_table_test = dataset_dict["lookup_table_test"]

        train_data = np.hstack((train_data, np.zeros([len(train_data), 1])))
        test_data = np.hstack((test_data, np.zeros([len(test_data), 1])))
        for key in list(lu_table_train.keys()):
            lu_table_train[key]["hdr"] = spectral.open_image(lu_table_train[key]["file_path"])
            train_data[lu_table_train[key]["min"]:lu_table_train[key]["max"], 2] = key
        for key in list(lu_table_test.keys()):
            lu_table_test[key]["hdr"] = spectral.open_image(lu_table_test[key]["file_path"])
            test_data[lu_table_test[key]["min"]:lu_table_test[key]["max"], 2] = key
        # return data_origin, data_train, targets_train, data_test, targets_test, meta
        return train_data.astype(int), lu_table_train, \
               test_data.astype(int), lu_table_test

    def load_data_to_array(self):
        data = self.rowcol
        data_list = list()
        label_list = list()
        for sample in tqdm(data):
            (row, col, key_lu_table) = sample
            img = self.lu_table[key_lu_table]["hdr"]
            sample = img[row, col].squeeze()[range(0, self.n_bands, 3)]
            np.nan_to_num(sample, copy=False)
            sample[sample > 1] = 1
            sample[sample < 0] = 0

            data_list.append(sample)
            label_list.append((row, col, key_lu_table))
        data_dict = dict()
        data_dict["x"] = data_list
        data_dict["y"] = label_list
        return data_dict

    def classification(self, labels, type="uv"):

        assert len(labels) == len(self.rowcol)

        for lu_table_key in list(self.lu_table.keys()):
            plt.figure(figsize=[15, 15])
            plt.subplot()
            plt.title(self.lu_table[lu_table_key]["file_path"])
            img = self.lu_table[lu_table_key]["hdr"]
            classification_of_file = labels[self.rowcol[:, 2] == lu_table_key]
            classification_of_file += 1  # zeros for background (not all the data of the imgis in the dataset)
            labels_of_file = np.zeros([img.shape[0], img.shape[1]])
            for idx, (row, col, _) in enumerate(self.rowcol[self.rowcol[:, 2] == lu_table_key]):
                labels_of_file[row][col] = classification_of_file[idx]
            rgb_bands = [514, 248, 155]
            if type == "uv":
                rgb_bands = [405, 229, 53]
            img_rgb = img.read_bands(rgb_bands)
            plt.imshow(img_rgb)
            plt.imshow(labels_of_file, alpha=1.)

        plt.show()


def save_model(state_dict, path):
    torch.save(state_dict, path.format("-crossval-0"))


def save_image(data, path):
    vutils.save_image(data, path.format("-crossval-0"),
                      normalize=True)


def test(load_to_mem):
    dataset = Hdr_dataset(load_to_mem=load_to_mem)
    dataloader = DataLoader(dataset, batch_size=256,
                            shuffle=True, num_workers=1, drop_last=True)

    time_start = time.time()
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())
        # if i_batch == 100:
    print(time.time() - time_start)


def show_samples(load_to_mem):
    dataset = Hdr_dataset(load_to_mem=load_to_mem)
    dataloader = DataLoader(dataset, batch_size=128,
                            shuffle=True, num_workers=8)

    for i_batch, sample_batched in enumerate(dataloader):
        sample_batched = sample_batched.data.cpu().numpy()
        for idx, x in enumerate(sample_batched):
            plt.plot(x, alpha=0.3)
        plt.show()


if __name__ == '__main__':
    test(load_to_mem=True)
    # show_samples(load_to_mem=True)
