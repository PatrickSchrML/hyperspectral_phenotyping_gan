import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import sys

sys.path.append('/home/patrick/repositories/hyperspec')
path = '/home/patrick/repositories/datasets/mirwaes/cerc15dai175.mat'  # TODO args


def build_dataset_only_ids(balance, train_ratio, n_splits, outf):
    outf += "_balanced" if balance else ""
    try:
        os.makedirs(outf)
    except OSError:
        pass

    data = scipy.io.loadmat(os.path.join(path))

    dim = data["dim"].reshape(-1)

    y = np.array(data["labels"].T, dtype=int)
    y -= 1
    y = y.squeeze()

    num_samples = y.shape[0]

    x = np.array(range(0, num_samples), dtype=int)
    x.reshape((-1, 1))

    #data_ids = dict()
    #idx = 0
    #for row_idx in tqdm(range(dim[0])):
    #    for col_idx in range(dim[1]):
    #        data_ids[idx] = (row_idx, col_idx)

    if train_ratio == 1:
        n_splits = 1
        x_balanced, _ = balanced_subsample(x, y)
        _dict = dict()
        _dict["train"] = x
        _dict["test"] = x
        _dict["train_balanced"] = np.array(x_balanced, dtype=int)
        _dict["test_balanced"] = np.array(x_balanced, dtype=int)
        pickle.dump(_dict, open(outf + "/split_{}.p".format(0), "wb"))
    else:
        # build cross validation splits
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 - train_ratio, random_state=1234)

        split_idx = 0
        for train_index, test_index in sss.split(x, y):
            train_index_balanced, _ = balanced_subsample(train_index, y[train_index])
            test_index_balanced, _ = balanced_subsample(test_index, y[test_index])
            _dict = dict()
            _dict["train"] = train_index
            _dict["test"] = test_index
            _dict["train_balanced"] = np.array(train_index_balanced, dtype=int)
            _dict["test_balanced"] = np.array(test_index_balanced, dtype=int)

            pickle.dump(_dict, open(outf + "/split_{}.p".format(split_idx), "wb"))
            split_idx += 1

    meta_dict = dict()
    meta_dict["num_samples"] = num_samples
    meta_dict["dim"] = dim
    meta_dict["num_classes"] = np.amax(y) + 1
    meta_dict["balanced"] = balance
    meta_dict["train_ratio"] = train_ratio
    meta_dict["n_splits"] = n_splits
    meta_dict["outf"] = outf
    meta_dict["data_path"] = path

    # save all the data
    pickle.dump(meta_dict, open(outf + "/meta.p", "wb"))


def build_dataset(balance, train_ratio, n_splits, outf):
    outf += "_balanced" if balance else ""
    try:
        os.makedirs(outf)
    except OSError:
        pass

    data = scipy.io.loadmat(os.path.join(path))

    dim = data["dim"].reshape(-1)

    x = data['counts']
    x = x.T

    num_samples = len(x)

    x = np.reshape(x, newshape=dim)
    x = x.astype(float)

    y = data["labels"]
    y = np.reshape(y, newshape=[dim[0], dim[1]])
    y = y.astype(int) - 1

    data_packed = np.empty([num_samples, 7], dtype=object)

    wavelength = np.array(data["wavelength"][0]).astype(int)
    dict_wavelength = dict()
    for idx_w, w in enumerate(wavelength):
        dict_wavelength[w] = idx_w

    idx = 0
    for row_idx in tqdm(range(dim[0])):
        for col_idx in range(dim[1]):
            data_packed[idx][0] = x[row_idx, col_idx]
            data_packed[idx][1] = wordify_pixel(x[row_idx, col_idx])
            data_packed[idx][2] = y[row_idx, col_idx]
            data_packed[idx][3] = row_idx
            data_packed[idx][4] = col_idx
            data_packed[idx][5] = idx
            data_packed[idx][6] = create_veg_idx(x[row_idx, col_idx], dict_wavelength)
            idx += 1
    data_packed_total = data_packed.copy()
    if balance:
        data_packed, _ = balanced_subsample(data_packed, data_packed[:, 2].astype(int))

    if train_ratio == 1:
        n_splits = 1
        pickle.dump(data_packed, open(outf + "/split_{}_train.p".format(0), "wb"))
        pickle.dump(data_packed, open(outf + "/split_{}_test.p".format(0), "wb"))
    else:
        # build cross validation splits
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=1 - train_ratio, random_state=1234)
        X_ = np.array(data_packed[:, 0].tolist(), dtype=float)
        y_ = data_packed[:, 2].astype(int)

        split_idx = 0
        for train_index, test_index in sss.split(X_, y_):
            train_set = data_packed[train_index]
            test_set = data_packed[test_index]

            pickle.dump(train_set, open(outf + "/split_{}_train.p".format(split_idx), "wb"))
            pickle.dump(test_set, open(outf + "/split_{}_test.p".format(split_idx), "wb"))
            split_idx += 1

    meta_dict = dict()
    meta_dict["num_samples"] = num_samples
    meta_dict["dim"] = dim
    meta_dict["num_classes"] = np.amax(y) + 1
    meta_dict["balanced"] = balance
    meta_dict["train_ratio"] = train_ratio
    meta_dict["n_splits"] = n_splits
    meta_dict["data_cols"] = "x, x wordified, label, row, col"
    meta_dict["outf"] = outf

    # save all the data
    pickle.dump(data_packed_total, open(outf + "/total.p", "wb"))
    pickle.dump(meta_dict, open(outf + "/meta.p", "wb"))


def get_data_by_idx(indices, meta):

    data = scipy.io.loadmat(os.path.join(meta["data_path"]))

    x = data['counts']
    x = x.T

    y = data["labels"]
    y = y.T
    y -= 1

    x = x.astype(float)
    y = y.astype(int)

    return x[indices], y[indices], indices


def create_veg_idx(data, wavelength):
    indices = []
    #  msr = (_gew(data, 750, wavelength) - _gew(data, 445, wavelength)) / \
    #      (_gew(data, 705, wavelength) + _gew(data, 445, wavelength)) # (750 - 445) / (705 + 445)
    ndvi = (_gew(data, 799, wavelength) - _gew(data, 669, wavelength)) / \
           (_gew(data, 799, wavelength) + _gew(data, 669, wavelength))  # (800 - 670) / (800 + 670)
    pri = (_gew(data, 530, wavelength) - _gew(data, 569, wavelength)) / \
          (_gew(data, 530, wavelength) + _gew(data, 569, wavelength))  # (531 - 570) / (531 + 570)
    #sipi = (_gew(data, 799, wavelength) - _gew(data, 445, wavelength)) / \
    #       (_gew(data, 799, wavelength) + _gew(data, 680, wavelength))  # (800 - 445) / (800 + 680)
    pssr_a = _gew(data, 799, wavelength, print_idx=False) / _gew(data, 680, wavelength, print_idx=False)  # (800 - 680)
    pssr_b = _gew(data, 799, wavelength) / _gew(data, 635, wavelength)  # (800 - 635)
    pssr_c = _gew(data, 799, wavelength) / _gew(data, 469, wavelength)  # (800 - 470)
    psnd_a = (_gew(data, 799, wavelength) - _gew(data, 680, wavelength)) / \
             (_gew(data, 799, wavelength) + _gew(data, 680, wavelength))  # (800 - 680) / (800 + 680)
    psnd_b = (_gew(data, 799, wavelength) - _gew(data, 635, wavelength)) / \
             (_gew(data, 799, wavelength) + _gew(data, 635, wavelength))  # (800 - 635) / (800 - 635)
    psnd_c = (_gew(data, 799, wavelength) - _gew(data, 469, wavelength)) / \
             (_gew(data, 799, wavelength) + _gew(data, 469, wavelength))  # (800 - 470) / (800 - 470)
    rsri = (_gew(data, 680, wavelength) - _gew(data, 499, wavelength)) / \
           (_gew(data, 750, wavelength))  # (680 - 500) / 750
    indices.append(ndvi)
    indices.append(pri)
    indices.append(pssr_a)
    indices.append(pssr_b)
    indices.append(pssr_c)
    indices.append(psnd_a)
    indices.append(psnd_b)
    indices.append(psnd_c)
    indices.append(rsri)
    indices = np.array(indices)
    return indices


# get_energie_of_wavelength
def _gew(data, wave_length, wavelength_dict, print_idx=False):
    if print_idx:
        print(wavelength_dict[wave_length])
    return data[wavelength_dict[wave_length]]


def get_data(path, split=0, type="train"):
    if type == "meta":
        data = pickle.load(open(path + "/meta.p", "rb"))
    elif type == "test":
        data = pickle.load(open(path + "/split_{}_test.p".format(split), "rb"))
    elif type == "train":
        data = pickle.load(open(path + "/split_{}_train.p".format(split), "rb"))
    elif type == "all":
        data = pickle.load(open(path + "/total.p", "rb"))
    else:
        raise ValueError("not supported type")

    return data


def get_idx_in_origin(data):
    return data[:, 5].astype(int)


def get_veg_indices(data):
    indices = np.array(data[:, 6].tolist(), dtype=float)
    return indices


def get_veg_indices_std(indices):
    indices_t = indices.T
    std_indices = np.std(indices_t, axis=1)
    #print(std_indices)
    return std_indices


def get_veg_indices_mean(indices):
    indices_t = indices.T
    mean_indices = np.mean(indices_t, axis=1)
    #print(mean_indices)
    return mean_indices


def get_veg_indices_min(indices):
    indices_t = indices.T
    min_indices = np.amin(indices_t, axis=1)
    return min_indices


def get_veg_indices_max(indices):
    indices_t = indices.T
    max_indices = np.amax(indices_t, axis=1)

    return max_indices


def norm_veg_indices(indices):
    min_indices = get_veg_indices_min(indices)
    max_indices = get_veg_indices_min(indices)
    indices_normed = (indices - min_indices) / max_indices
    return indices_normed


def norm_veg_indices_by_data(indices, data):
    min_indices = get_veg_indices_min(data)
    max_indices = get_veg_indices_max(data)
    indices_normed = (indices - min_indices) / max_indices

    return indices_normed


def get_x_y_pos(data):
    #print(data.size) # in helper test = train
    return np.array(data[:, 0].tolist(), dtype=float), data[:, 2].astype(int), data[:, 3:5].astype(int)


def get_x_y_pos_veg(data):
    return np.array(data[:, 0].tolist(), dtype=float), data[:, 2].astype(int), data[:, 3:5].astype(int), np.array(data[:, 6].tolist(), dtype=float)


def get_x_wordified_y_pos(data):
    return np.array(data[:, 1].tolist(), dtype=str), data[:, 2].astype(int), data[:, 3:5].astype(int)


def get_reshaped(data, pos, meta):
    dim = meta["dim"]
    positions_in_img_flatten = np.array([(row * dim[1]) + col for (row, col) in pos])
    positions_in_img_inds = positions_in_img_flatten.argsort()
    data_copy = data.copy()
    data_sorted = data_copy[positions_in_img_inds]
    data_reshaped = np.reshape(data_sorted, newshape=[dim[0], dim[1], -1])
    return data_reshaped


def get_pos(data):
    return data[:, 3:5].astype(int)


def balanced_subsample(x, y, subsample_size=1.0):
    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[y == yi]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems * subsample_size)

    xs = []
    ys = []

    for ci, this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs, ys


def print_as_img_without_reshape(labels, save_path=None):
    predictions_reshaped = labels
    plt.imshow(predictions_reshaped, cmap="tab10")
    plt.axis("off")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")


def print_as_img_without_sort(labels, meta, save_path=None):
    dim = meta["dim"]
    predictions_reshaped = np.reshape(labels, newshape=[dim[0], dim[1]])
    plt.imshow(predictions_reshaped, cmap="tab10")
    plt.axis("off")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")


def print_as_img(labels, pos, meta, save_path=None):
    dim = meta["dim"]
    positions_in_img_flatten = np.array([(row * dim[1]) + col for (row, col) in pos])
    positions_in_img_inds = positions_in_img_flatten.argsort()
    #print(positions_in_img_inds.shape)
    predictions_out_copy = labels.copy()
    predictions_sorted = predictions_out_copy[positions_in_img_inds]
    predictions_reshaped = np.reshape(predictions_sorted, newshape=[dim[0], dim[1]])
    plt.imshow(predictions_reshaped, cmap="tab10")
    plt.axis("off")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, transparent=True, bbox_inches="tight")


def wordify_pixel(pixel, start_band=0, end_band=1000):
    # pool = multiprocessing.Pool(processes=16)

    # def my_func(i, d):
    #    return "w%de%d" % (d.astype(float) * 100, i)

    # sentence = np.empty(data.shape[2], dtype=object)
    # print(enumerate(data.read_pixel(row, col)))
    # 1 / 0
    # sentence[:] = pool.map(my_func, (t in enumerate(data.read_pixel(row, col))))

    sentence = []
    _data = pixel
    for idx, word_raw in enumerate(_data):
        if start_band <= idx <= end_band:
            word_raw = round(word_raw, 1)
            sentence.append(("w%de%d" % (idx, word_raw * 100)))
    return sentence

###
