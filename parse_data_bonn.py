from spectral import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from multiprocessing import Pool
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import OrderedDict

plt.figure(figsize=(15, 15))

data_path_vnir = [
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag1_KeinNaCL_20gl_80gl/VNIRHyperspecCamera_G4_447/Tag1_KeinNaCL_20gl_80gl_2018_05_15_12_14_08/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag1_KeinNaCL_20gpl_80gpl/VNIRHyperspecCamera_G4_447/Tag1_KeinNaCL_20gpl_80gpl_2018_05_15_10_20_33/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag2_keinNaCl_20gpl_80gpl/VNIRHyperspecCamera_G4_447/Tag2_keinNaCl_20gpl_80gpl_2018_05_16_10_28_54/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag2_keinNaCl_20gpl_80gpl/VNIRHyperspecCamera_G4_447/Tag2_keinNaCl_20gpl_80gpl_2018_05_16_11_37_12/raw_norm.hdr',
    ""
    # '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag3_keinNaCl_20mgpl_80mgpl/VNIRHyperspecCamera_G4_447/Tag3_keinNaCl_20mgpl_80mgpl_2018_05_17_10_17_40/raw_norm.hdr'
    ,
    ""
    # '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag3_keinNaCl_20mgpl_80mgpl/VNIRHyperspecCamera_G4_447/Tag3_keinNaCl_20mgpl_80mgpl_2018_05_17_11_26_55/raw_norm.hdr'
    ,
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag4_keinNaCll_20gpl_80gpl/VNIRHyperspecCamera_G4_447/Tag4_keinNaCll_20gpl_80gpl_2018_05_18_09_28_54/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag4_keinNaCll_20gpl_80gpl/VNIRHyperspecCamera_G4_447/Tag4_keinNaCll_20gpl_80gpl_2018_05_18_10_29_48/raw_norm.hdr'
]

data_path_uv = [
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag1_KeinNaCL_20gl_80gl/UVHyperspecCamera_G4_419/Tag1_KeinNaCL_20gl_80gl_2018_05_15_12_04_12/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag1_KeinNaCL_20gpl_80gpl/UVHyperspecCamera_G4_419/Tag1_KeinNaCL_20gpl_80gpl_2018_05_15_09_56_53/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag2_keinNaCl_20gpl_80gpl/UVHyperspecCamera_G4_419/Tag2_keinNaCl_20gpl_80gpl_2018_05_16_10_18_59/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag2_keinNaCl_20gpl_80gpl/UVHyperspecCamera_G4_419/Tag2_keinNaCl_20gpl_80gpl_2018_05_16_11_27_17/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag3_keinNaCl_20mgpl_80mgpl/UVHyperspecCamera_G4_419/Tag3_keinNaCl_20mgpl_80mgpl_2018_05_17_10_07_47/raw_norm.hdr',
    "",
    # '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag3_keinNaCl_20mgpl_80mgpl/UVHyperspecCamera_G4_419/Tag3_keinNaCl_20mgpl_80mgpl_2018_05_17_11_17_02/raw_norm.hdr'
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag4_keinNaCll_20gpl_80gpl/UVHyperspecCamera_G4_419/Tag4_keinNaCll_20gpl_80gpl_2018_05_18_09_19_03/raw_norm.hdr',
    '/media/disk2/datasets/hyperspectral_data_anna_bonn/Tag4_keinNaCll_20gpl_80gpl/UVHyperspecCamera_G4_419/Tag4_keinNaCll_20gpl_80gpl_2018_05_18_10_19_57/raw_norm.hdr'
]


def find_hdr_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("data.hdr"):
                print(os.path.join(root, file))


def parse_single_file_vnir(path_and_idx):
    path = path_and_idx[0]
    if path == "":
        return
    idx = path_and_idx[1]
    max_row = 700
    if idx == 1:
        max_row = 1030
    img = open_image(path)
    n_row = max_row  # img.shape[0]
    n_col = img.shape[1]
    labels = np.zeros([n_row, n_col])
    for row in tqdm(range(n_row)):
        for col in range(n_col):
            pixel = img.read_subimage([row], [col], [250, 448, 772])  # img[row, col]
            pixel = pixel.squeeze()
            # if pixel[250] < 0.2 and pixel[448] < 0.15 and pixel[772] < 0.3 and pixel[772] > 0.15:
            if pixel[0] < 0.2 and pixel[1] < 0.15 and pixel[2] < 0.3 and pixel[2] > 0.15:
                labels[row, col] = 1
                # plt.plot(pixel)
                # plt.show()
    parsed_data = dict()
    parsed_data["path"] = path
    parsed_data["labels"] = labels
    pickle.dump(parsed_data,
                open("/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/vnir/file{}.p".format(idx),
                     "wb"))
    print("Finished File", idx, path)


def parse_single_file_uv_kmeans(path_and_idx):
    path = path_and_idx[0]
    if path == "":
        return
    idx = path_and_idx[1]
    img = open_image(path)
    img_np = img[75:530, 85:1190, :]
    dim = img_np.shape
    img_np = np.reshape(img_np, [-1, dim[2]])
    #test = np.isnan(img_np)
    #print(len(test[test == True]))
    #1 / 0
    np.nan_to_num(img_np, copy=False)
    kmeans = KMeans(n_clusters=20, random_state=0, n_jobs=-6, algorithm="elkan", max_iter=10000).fit(img_np)
    labels = kmeans.labels_
    labels = np.reshape(labels, [dim[0], dim[1]])

    rgb_bands = [405, 229, 53]  # [514, 248, 155]
    img_rgb = img.read_bands(rgb_bands)
    img_rgb = img_rgb[75:530, 85:1190, :]
    plt.imshow(img_rgb)
    plt.imshow(labels, alpha=1)
    plt.show()
    1 / 0


def parse_single_file_uv(path_and_idx):
    path = path_and_idx[0]
    if path == "":
        return
    idx = path_and_idx[1]
    max_row = 700
    if idx == 1:
        max_row = 1030
    img = open_image(path)
    # print(img.shape)
    #pixel = img[143, 354]  # img[row, col]
    #start = img.read_subimage([143], [354], range(10, 40)).mean()
    #print(start)
    #plt.plot(pixel)
    #plt.show()
    #1 / 0

    # start = img.read_subimage([143], [354], [10]).mean()
    # first = img.read_subimage([143], [354], range(190, 211)).mean() # [143], [354] ; [300], [420]
    # second = img.read_subimage([143], [354], range(240, 261)).mean()
    # print(start / first) # 1.6
    # print(second / first)  # 1.08, 1.1745734
    # 1 / 0
    n_row = max_row  # img.shape[0]
    n_col = img.shape[1]
    labels = np.zeros([n_row, n_col])
    for row in tqdm(range(n_row)):
        for col in range(n_col):
            start = img.read_subimage([row], [col], range(10, 40)).mean()
            first = img.read_subimage([row], [col], range(190, 211)).mean()
            second = img.read_subimage([row], [col], range(240, 261)).mean()
            ratio1 = start / first
            ratio2 = second / first
            # if start > 0.25 and ratio1 > 1.5 and 1. < ratio2 < 1.3: works
            if 0.25 < start < 1. and ratio1 > 1.4 and 1. < ratio2 < 1.3:
                labels[row, col] = 1
                # plt.plot(img[row, col])
                # plt.show()
                #print(row, col)
    parsed_data = dict()
    parsed_data["path"] = path
    parsed_data["labels"] = labels
    pickle.dump(parsed_data,
                open("/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/uv/file{}.p".format(idx),
                     "wb"))
    print("Finished File", idx, path)
    #1 /0


def parse_mp(type=None):
    if type != "vnir":
        print("Start parsing uv data")
        data_path_and_idx = list(zip(data_path_uv, range(0, len(data_path_uv))))
        p = Pool(np.minimum(len(data_path_uv), 28))
        p.map(parse_single_file_uv, data_path_and_idx)
        print("Finished parsing uv data")

    if type != "uv":
        print("Start parsing vnir data")
        data_path_and_idx = list(zip(data_path_vnir, range(0, len(data_path_vnir))))
        p = Pool(np.minimum(len(data_path_vnir), 28))
        p.map(parse_single_file_vnir, data_path_and_idx)
        print("Finished parsing vnir data")


def parse(type=None):
    if type != "vnir":
        print("Start parsing uv data")
        for idx, path in enumerate(data_path_uv):
            parse_single_file_uv((path, idx))
        print("Finished parsing uv data")

    if type != "uv":
        print("Start parsing vnir data")
        for idx, path in enumerate(data_path_vnir):
            parse_single_file_vnir((path, idx))
        print("Finished parsing vnir data")


def parse_kmeans(type=None):
    if type != "vnir":
        print("Start parsing uv data")
        for idx, path in enumerate(data_path_uv):
            parse_single_file_uv_kmeans((path, idx))
        print("Finished parsing uv data")

    if type != "uv":
        print("Start parsing vnir data")
        for idx, path in enumerate(data_path_vnir):
            parse_single_file_vnir((path, idx))
        print("Finished parsing vnir data")


def show_leafs(idx, type="vnir"):
    data = pickle.load(
        open("/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/{}/file{}.p".format(type, idx),
             "rb"))
    path = data["path"]
    labels = data["labels"]
    print("File:", path, "; Number of plant pixel:", len(labels[labels == 1]))
    img = open_image(path)
    rgb_bands = [514, 248, 155]
    if type == "uv":
        rgb_bands = [405, 229, 53]
    img_rgb = img.read_bands(rgb_bands)
    plt.imshow(img_rgb)
    plt.imshow(labels, alpha=0.6)

    plt.show()


def _get_leaf_pixel(path_labels, type="vnir"):
    #path_labels = "/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/{}/file{}.p".format(type, idx)
    data = pickle.load(
        open(path_labels, "rb"))
    path = data["path"]
    if path == "":
        return
    labels = data["labels"]
    img = open_image(path)

    leaf_pixels = list()

    test = labels.copy()
    test = test.reshape(-1)
    #print("Num Pixel in image:", len(test))

    for row in range(labels.shape[0]):
        for col in range(labels.shape[1]):
            if labels[row, col] == 1:
                leaf_pixels.append((img.read_pixel(row, col), row, col))

    leaf_pixels = np.array(leaf_pixels)
    #print("Num Pixel of leafs in image", len(leaf_pixels))

    data_dict = dict()
    data_dict["data"] = leaf_pixels
    data_dict["path"] = path

    return data_dict


def save_leaf_pixel(type="vnir"):
    path_dir = "/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/{}".format(type)
    path_output = "/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leaf_pixel/{}".format(type)
    path_files = list()
    for file_name in os.listdir(path_dir):
        if file_name.endswith(".p"):
            path_files.append((os.path.join(path_dir, file_name), file_name))

    for path_labels, file_name in tqdm(path_files):
        data = _get_leaf_pixel(path_labels, type)
        pickle.dump(data, open(os.path.join(path_output, file_name), "wb"))


def delete_not_normalized(img, mapping):

    new_mapping = list()
    for m in tqdm(mapping):
        sample = img[m[0], m[1]].squeeze()
        np.nan_to_num(sample, copy=False)
        if not (np.any((sample > 1.)) or np.any((sample < 0.))):
            new_mapping.append(m)

    new_mapping = np.array(new_mapping)
    return new_mapping


def _convert_to_idx(labeled_img):
    tmp = labeled_img.copy().reshape(-1)
    indices_img_pos = np.zeros([len(tmp[tmp == 1]), 2], dtype=int)
    current_idx = 0
    for row in range(labeled_img.shape[0]):
        # TODO Labels
        for col in range(labeled_img.shape[1]):
            if labeled_img[row,  col] == 1:
                indices_img_pos[current_idx][0] = row
                indices_img_pos[current_idx][1] = col
                current_idx += 1
    return indices_img_pos


def split_to_cross_val(type="vnir"):
    path_dir = "/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/leafs/{}".format(type)
    path_files = list()

    for file_name in os.listdir(path_dir):
        if file_name.endswith(".p"):
            path_files.append((os.path.join(path_dir, file_name), file_name))

    current_idx_train = 0
    current_idx_test = 0
    lu_table_train = OrderedDict()  # TODO train and test table
    lu_table_test = OrderedDict()  # TODO train and test table

    train_data = np.empty([0, 2], dtype=int)
    test_data = np.empty([0, 2], dtype=int)
    for path_labels, file_name in tqdm(path_files):
        # load background/plant labels
        if file_name == "file0.p" or file_name == "file7.p":  # just use one image of day 1 and one image of day 4

            data = pickle.load(
                open(path_labels, "rb"))
            path_hdr = data["path"]
            if path_hdr == "":
                break
            labels = data["labels"]
            # seperate plant data from background and get mapping to image position
            map_idx_to_colrow = _convert_to_idx(labels)
            print(len(map_idx_to_colrow))
            map_idx_to_colrow = delete_not_normalized(open_image(path_hdr), map_idx_to_colrow)
            print(len(map_idx_to_colrow))
            # split data to train and test
            train_split, test_split = train_test_split(map_idx_to_colrow, train_size=0.1, random_state=0)

            train_data = np.vstack((train_data, train_split))
            test_data = np.vstack((test_data, test_split))

            # update lookup table with current data
            count_train = len(train_split)
            count_test = len(test_split)

            lu_table_train[current_idx_train] = {"min": current_idx_train,
                                                 "max": current_idx_train+count_train,
                                                 "count": count_train,
                                                 "file_path": path_hdr}
            current_idx_train += current_idx_train+count_train

            lu_table_test[current_idx_test] = {"min": current_idx_test,
                                               "max": current_idx_test + count_test,
                                               "count": count_test,
                                               "file_path": path_hdr}
            current_idx_test += current_idx_test + count_test

    print("Num training data:", len(train_data))

    # save train and test data
    data = dict()
    data["train"] = train_data
    data["test"] = test_data
    data["lookup_table_train"] = lu_table_train
    data["lookup_table_test"] = lu_table_test
    pickle.dump(data, open("/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/dataset/dataset_small_10prozent.p", "wb"))


def test_splits():
    dataset_dict = pickle.load(
        open("/media/disk2/datasets/hyperspectral_data_anna_bonn/parsed_data/dataset/dataset_small_10prozent.p", "rb"))
    train_data = dataset_dict["train"]
    test_data = dataset_dict["test"]
    lu_table_train = dataset_dict["lookup_table_train"]
    lu_table_test = dataset_dict["lookup_table_test"]

    #sample_mapping = train_data[0]
    #img = open_image(lu_table_train[0]["file_path"])

    #plt.plot(img[sample_mapping[0], sample_mapping[1]].squeeze())
    #plt.show()
    train_data = np.hstack((train_data, np.zeros([len(train_data), 1])))
    test_data = np.hstack((test_data, np.zeros([len(test_data), 1])))

    ids = list()
    for key in list(lu_table_train.keys()):
        lu_table_train[key]["hdr"] = open_image(lu_table_train[key]["file_path"])
        train_data[lu_table_train[key]["min"]:lu_table_train[key]["max"], 2] = key
        print(lu_table_train[key]["min"])
        ids.append(lu_table_train[key]["min"])
    for key in list(lu_table_test.keys()):
        lu_table_test[key]["hdr"] = open_image(lu_table_test[key]["file_path"])
        test_data[lu_table_test[key]["min"]:lu_table_test[key]["max"], 2] = key

    train_data = train_data.astype(int)

    for img_id in ids:
        for sample_mapping in train_data[train_data[:, 2] == img_id]:
            img = lu_table_train[sample_mapping[2]]["hdr"]
            sample = img[sample_mapping[0], sample_mapping[1]].squeeze()[range(0, 442, 3)]

            if np.any((sample > 1.)) or np.any((sample < 0.)):
                raise ValueError("signature values not between 0 and 1")

            plt.plot(sample, alpha=0.3)
        plt.show()

        train_data_plt = train_data[train_data[:, 2] == img_id]
        show_labels(lu_table_train[img_id]["hdr"], train_data_plt)


def show_labels(img, mappings, type="uv"):
    labels = np.zeros([img.shape[0], img.shape[1]])

    for mapping in mappings:
        labels[mapping[0], mapping[1]] = 1

    rgb_bands = [514, 248, 155]
    if type == "uv":
        rgb_bands = [405, 229, 53]
    img_rgb = img.read_bands(rgb_bands)
    plt.imshow(img_rgb)
    plt.imshow(labels, alpha=0.6)

    plt.show()


"""
    - call parse_mp to segmentate data
    - call show_leafs to plot segmentation
    - call save_leaf_pixel to save segmentation to pickled numpy array
    - call split_to_cross_val to split segmentation in test and train sets. Save mapping of row, col, img to disk
    - call test_splits to show the splitted data of train-set
"""
if __name__ == '__main__':
    #parse(type="uv")
    #parse_mp(type="uv")
    # idx = 5
    #show_leafs(7, type="uv")
    #save_leaf_pixel(type="uv")
    #split_to_cross_val("uv")
    #parse_kmeans(type="uv")
    #test_splits()
    find_hdr_files("/media/disk2/datasets/hyperspectral_data_anna_bonn_Salzstress")
