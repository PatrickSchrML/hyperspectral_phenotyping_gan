""""
Here we implement a class for loading data.
"""

import pickle
import torch
import numpy as np
import random
import os
import scipy.io
import sys
import torchvision.utils as vutils
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append('/home/patrick/repositories/hyperspec')
from data_helper.helper import get_x_y_pos, get_x_y_pos_veg, get_data, get_data_by_idx

np.random.seed(0)


def load_dataset_train(filename):
    return filename


def normalize(data, max_value=None):
    # data -= np.min(data)
    if max_value is None:
        max_value = np.max(data)
    data /= max_value
    return data


class DataLoader:
    def __init__(self, batch_size=128, sup_ratio=1., train_ratio=0.25, balanced=True):
        if train_ratio == 1:
            n_splits = 1
        else:
            n_splits = 2

        data_train, targets_train, data_test, targets_test, complete_x, complete_y, \
        meta = self.load_leaf(train_ratio, n_splits, balanced)

        self.x_train = data_train
        self.y_train = targets_train
        # compute which labels should be user for semi-supervised learning - by a mask
        if sup_ratio < 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sup_ratio, random_state=1234)
            _, y_sup_mask_in = next(sss.split(self.x_train, self.y_train))
            y_sup_mast_indices = np.array(range(len(self.y_train)))
            self.y_sup_mask = np.in1d(y_sup_mast_indices, y_sup_mask_in)
        else:
            self.y_sup_mask = np.array([True for _ in range(len(self.y_train))])

        self.x_test = data_test
        self.y_test = targets_test

        self.data_train = list(zip(data_train, targets_train))
        self.data_total = list(zip(complete_x, complete_y))

        self.num_classes = np.amax(targets_train).astype(int) + 1
        self.size_train = len(self.x_train)
        self.size_test = len(self.x_test)

        self.meta = meta
        self.currentIndex = 0
        self.currentIndex_testset = 0

        if batch_size is None:
            self.batch_size = self.size_train
        self.batch_size = batch_size

    def load_leaf(self, train_ratio, n_splits, balanced=True):
        return self.load_leaf_by_dataset_idx(train_ratio, n_splits, balanced)

    def load_leaf_by_dataset_idx(self, train_frac, n_splits, balanced=True):
        split = 0
        path = '/home/patrick/repositories/hyperspec/data_helper/datasets_ids/dataset_train-ratio-{}%_n-splits-{}'.format(
            int(train_frac * 100), n_splits)
        meta = pickle.load(open(path + "/meta.p", "rb"))
        data = pickle.load(open(path + "/split_{}.p".format(split), "rb"))
        # switch train and test data -> so trainset is bigger then testset
        #test_x, test_y = get_data_by_idx(data["test"], meta)
        #train_x, train_y = get_data_by_idx(data["train_balanced" if balanced else "train"], meta)
        test_x, test_y = get_data_by_idx(data["train"], meta)
        train_x, train_y = get_data_by_idx(data["test_balanced" if balanced else "test"], meta)

        complete_x, complete_y = get_data_by_idx(np.hstack((data["train"], data["test"])), meta)

        max_value = np.maximum(np.max(test_x), np.max(train_x))

        test_x = normalize(test_x, max_value=max_value)  # normalize data to be between [0,1]
        train_x = normalize(train_x, max_value=max_value)  # normalize data to be between [0,1]

        # return data_origin, data_train, targets_train, data_test, targets_test, meta
        return train_x, train_y, test_x, test_y, complete_x, complete_y, meta

    # just used by old implementation - dont remove
    def load_leaf_by_dataset(self, train_frac, n_splits):

        path = '/home/patrick/repositories/hyperspec/data_helper/datasets/dataset_train-ratio-{}%_n-splits-{}'.format(
            int(train_frac * 100), n_splits)
        meta = get_data(path, type="meta")

        # switch train and test data -> so trainset is bigger then testset
        test_x, test_y, _ = get_x_y_pos_veg(get_data(path, type="train", split=0))
        train_x, train_y, _ = get_x_y_pos_veg(get_data(path, type="test", split=0))

        max_value = np.maximum(np.max(test_x), np.max(train_x))

        data = train_x
        input_mat_gt = train_y

        data = normalize(data, max_value=max_value)  # normalize data to be between [0,1]
        class0_x, class0_y = data[input_mat_gt == 0], input_mat_gt[input_mat_gt == 0]
        class1_x, class1_y = data[input_mat_gt == 1], input_mat_gt[input_mat_gt == 1]
        class2_x, class2_y = data[input_mat_gt == 2], input_mat_gt[input_mat_gt == 2]

        num_sample_of_each_class = np.minimum(len(class0_y), np.minimum(len(class1_y), len(class2_y)))
        indices = np.array(range(0, num_sample_of_each_class))
        np.random.shuffle(indices)
        class0_x, class0_y = class0_x[indices], class0_y[indices]
        class1_x, class1_y = class1_x[indices], class1_y[indices]
        class2_x, class2_y, = class2_x[indices], class2_y[indices]

        # print(class0_x.shape)
        # print(class1_x.shape)
        # print(class2_x.shape)

        data_train = np.append(class0_x, np.append(class1_x, class2_x, axis=0), axis=0)
        targets_train = np.append(class0_y, np.append(class1_y, class2_y, axis=0), axis=0)

        data_test = normalize(test_x, max_value=max)
        targets_test = test_y

        # return data_train, targets_train, data_test, targets_test, meta
        return data_train, targets_train, data_test, targets_test, meta

    def fetch_batch(self, onehot=False, num_classes=-1):
        if self.currentIndex >= self.size_train - self.batch_size:
            self.currentIndex = 0

        if self.currentIndex == 0:
            shuffled_indices = np.array(range(0, self.size_train))
            np.random.shuffle(shuffled_indices)
            self.x_train = self.x_train[shuffled_indices]
            self.y_train = self.y_train[shuffled_indices]
            self.y_sup_mask = self.y_sup_mask[shuffled_indices]

        first_index = self.currentIndex
        self.currentIndex += self.batch_size

        batch_x = self.x_train[first_index:first_index + self.batch_size]
        batch_y = self.y_train[first_index:first_index + self.batch_size]
        batch_sup_indices = np.array(range(len(batch_y)))[
            self.y_sup_mask[first_index:first_index + self.batch_size] == True]

        batch_x = torch.FloatTensor(batch_x)

        if onehot:
            if num_classes == -1:
                num_classes = self.num_classes
            batch_y_ = torch.FloatTensor(self.batch_size, num_classes)
            batch_y_.zero_()
            batch_y_.scatter_(1, torch.LongTensor(batch_y).view(-1, 1), 1)
        else:
            batch_y_ = torch.FloatTensor(batch_y)

        return batch_x, batch_y_, torch.LongTensor(batch_sup_indices)

    def fetch_batch_splited_labels(self):
        if self.currentIndex >= self.size_train - self.batch_size:
            self.currentIndex = 0

        if self.currentIndex == 0:
            shuffled_indices = np.array(range(0, self.size_train))
            np.random.shuffle(shuffled_indices)
            self.x_train = self.x_train[shuffled_indices]
            self.y_train = self.y_train[shuffled_indices]
            self.y_sup_mask = self.y_sup_mask[shuffled_indices]

        first_index = self.currentIndex
        self.currentIndex += self.batch_size

        batch_x = self.x_train[first_index:first_index + self.batch_size]
        batch_y = self.y_train[first_index:first_index + self.batch_size]
        batch_sup_indices = np.array(range(len(batch_y)))[
            self.y_sup_mask[first_index:first_index + self.batch_size] == True]

        # split into 2 one-hot encodings
        batch_y_0 = batch_y.copy()  # batch_y == 0 or batch_y == 2, gesund blatt und gesund stamm
        batch_y_0[batch_y_0 == 1] = 0  # merge ges. blatt und krankes blatt -> label 0
        batch_y_0[batch_y_0 == 2] = 1  # stamm -> label 1
        batch_y_1 = batch_y.copy()
        batch_y_1[batch_y_1 == 2] = 0  # merge ges. blatt und ges. stamm -> label 0, krank: label 1

        batch_y_0_ = torch.FloatTensor(len(batch_y_0), 2)
        batch_y_0_.zero_()
        batch_y_0_.scatter_(1, torch.LongTensor(batch_y_0).view(-1, 1), 1)

        batch_y_1_ = torch.FloatTensor(len(batch_y_1), 2)
        batch_y_1_.zero_()
        batch_y_1_.scatter_(1, torch.LongTensor(batch_y_1).view(-1, 1), 1)

        batch_y_ = torch.cat((batch_y_0_, batch_y_1_), dim=1)

        # create tensor for batch x
        batch_x = torch.FloatTensor(batch_x)

        return batch_x, batch_y_, torch.LongTensor(batch_sup_indices)

    def fetch_batch_splited_labels_onehot_and_conti(self):
        if self.currentIndex >= self.size_train - self.batch_size:
            self.currentIndex = 0

        if self.currentIndex == 0:
            shuffled_indices = np.array(range(0, self.size_train))
            np.random.shuffle(shuffled_indices)
            self.x_train = self.x_train[shuffled_indices]
            self.y_train = self.y_train[shuffled_indices]
            self.y_sup_mask = self.y_sup_mask[shuffled_indices]

        first_index = self.currentIndex
        self.currentIndex += self.batch_size

        batch_x = self.x_train[first_index:first_index + self.batch_size]
        batch_y = self.y_train[first_index:first_index + self.batch_size]
        batch_sup_indices = np.array(range(len(batch_y)))[
            self.y_sup_mask[first_index:first_index + self.batch_size] == True]

        # split into 2 one-hot encodings
        batch_y_0 = batch_y.copy()  # batch_y == 0 or batch_y == 2, gesund blatt und gesund stamm
        batch_y_0[batch_y_0 == 1] = 0  # merge ges. blatt und krankes blatt -> label 0
        batch_y_0[batch_y_0 == 2] = 1  # stamm -> label 1
        batch_y_1 = batch_y.copy()
        batch_y_1[batch_y_1 == 0] = -1
        batch_y_1[batch_y_1 == 2] = -1  # merge ges. blatt und ges. stamm -> label 0, krank: label 1

        batch_y_0_ = torch.FloatTensor(len(batch_y_0), 2)
        batch_y_0_.zero_()
        batch_y_0_.scatter_(1, torch.LongTensor(batch_y_0).view(-1, 1), 1)

        batch_y_1_ = torch.FloatTensor(batch_y_1)

        # create tensor for batch x
        batch_x = torch.FloatTensor(batch_x)

        return batch_x, batch_y_0_, batch_y_1_, torch.LongTensor(batch_sup_indices)

    def fetch_batch_test(self):
        if self.currentIndex_testset >= self.size_test - self.batch_size:
            self.currentIndex_testset = 0

        first_index = self.currentIndex_testset
        self.currentIndex_testset += self.batch_size

        batch_x = self.x_test[first_index:first_index + self.batch_size]
        batch_y = self.y_test[first_index:first_index + self.batch_size]

        batch_x = torch.FloatTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)

        return batch_x, batch_y

    def fetch_samples(self, num_sample_each_class=1):
        # data, targets, _ = zip(*self.data_original)
        data, targets = zip(*self.data_train)
        data = np.array(data)
        targets = np.array(targets).squeeze()

        shuffled_indices = np.array(range(0, len(data)))
        np.random.shuffle(shuffled_indices)

        data = data[shuffled_indices]
        targets = targets[shuffled_indices]

        x = None
        y = None
        for i_class in range(self.num_classes):
            tmp_x, tmp_y = data[targets == i_class][:num_sample_each_class], \
                           targets[targets == i_class][:num_sample_each_class]
            if x is None:
                x = tmp_x
                y = tmp_y
            else:
                x = np.append(x, tmp_x, axis=0)
                y = np.append(y, tmp_y, axis=0)

        batch_x = np.array(x).reshape([num_sample_each_class * self.num_classes, 160])
        batch_y = np.array(y).reshape([num_sample_each_class * self.num_classes])

        batch_x = torch.FloatTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)

        return batch_x, batch_y

    def fetch_samples_mean(self):
        # data, targets, _ = zip(*self.data_original)
        data, targets = zip(*self.data_train)
        data = np.array(data)
        targets = np.array(targets).squeeze()
        x = None
        y = None
        for i_class in range(self.num_classes):
            tmp_x, tmp_y = np.mean(data[targets == i_class], axis=0), \
                           np.reshape(targets[targets == i_class][0], newshape=[-1, 1])
            if x is None:
                x = tmp_x
                y = tmp_y
            else:
                x = np.append(x, tmp_x, axis=0)
                y = np.append(y, tmp_y, axis=0)

        batch_x = np.array(x).reshape([self.num_classes, 160])
        batch_y = np.array(y).reshape([self.num_classes])

        batch_x = torch.FloatTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)

        return batch_x, batch_y

    def save_model(self, state_dict, path):
        torch.save(state_dict, path.format("-crossval-0"))

    def save_image(self, data, path):
        vutils.save_image(data, path.format("-crossval-0"),
                          normalize=True)

if __name__ == '__main__':
    data_loader = DataLoader(128, 0.1, train_ratio=0.25)
    print(data_loader.size_test)
    print(data_loader.size_train)