from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import argparse
from matplotlib.colors import hsv_to_rgb

style.use("ggplot")

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False, help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False, help='')
parser.add_argument('--n_dis', default=1, required=False, help='')
parser.add_argument('--n_noise', default=10, required=False, help='')
parser.add_argument('--outf_suffix', default="", required=False, help='')
parser.add_argument('--dataset', default="mat", help='mat or hdr')
parser.add_argument('--epoch', required=True, help='epoch...')

opt = parser.parse_args()


def plot_conti_code_tsne():
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments_{}/generated_code_noise{}_disc{}_conti{}_epoch{}.p".format(
            opt.dataset,
            opt.n_noise,
            opt.n_dis,
            opt.n_conti,
            opt.epoch),
        "rb"))
    labels = np.array(data["y"]).squeeze()
    labels_unique = np.unique(labels)
    code = np.array(data["z"]).copy()
    z = np.array(data["z"]).copy()
    # print(code[0])
    # code = code[:, -5:-2]
    code = code[:, -2:]
    # print(code[0])
    # 1 / 0
    signatures = np.array(data["x"])
    tsne = TSNE(n_jobs=26, n_components=2, learning_rate=100)
    Y = tsne.fit_transform(code)

    colors = ["red", "green", "blue"]
    for idx, label in enumerate(labels_unique):
        data_tsne = Y[labels == label]
        plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=colors[idx], alpha=0.3,
                    label=str(label))
    plt.legend()
    plt.show()


def plot_conti_code_as_img():
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_code_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
            opt.dataset,
            opt.nc,
            opt.n_dis,
            opt.n_conti,
            opt.n_noise,
            opt.epoch,
            opt.outf_suffix),
        "rb"))

    labels = np.array(data["y"]).squeeze()
    real = np.array(data["x"])
    code = np.array(data["z"]).copy()

    idx_in_img = np.array(data["origin_indices_in_img"])

    meta = data["meta"]
    dim = meta["dim"]

    # tmp = code.copy()
    # tmp[:, 2] = 0.5
    # code[:, 2] = 1
    # code[labels == 1] = tmp[labels == 1]
    # img = np.zeros([dim[0], dim[1], 3])
    sorted_labels = np.array([x for _, x in sorted(zip(idx_in_img, labels))])
    sorted_code = np.array([x for _, x in sorted(zip(idx_in_img, code))])
    sorted_real = np.array([x for _, x in sorted(zip(idx_in_img, real))])

    sorted_real = np.vstack((np.vstack((sorted_real[:, 59],
                                        sorted_real[:, 23])),
                             sorted_real[:, 5])).T

    sorted_code_normed = np.zeros_like(sorted_code, dtype=int)
    for idx in range(sorted_code.shape[1]):
        tmp = sorted_code[:, idx].copy()
        min_tmp = np.amin(tmp)
        tmp -= min_tmp
        max_tmp = np.amax(tmp)
        if max_tmp != 0:
            tmp /= max_tmp
        tmp *= 255
        sorted_code_normed[:, idx] = tmp.astype(int)

    img_labels = np.reshape(sorted_labels, newshape=[dim[0], dim[1]])
    img_code = np.reshape(sorted_code_normed, newshape=[dim[0], dim[1], 3])
    img_real = np.reshape(sorted_real, newshape=[dim[0], dim[1], 3])

    # print(img_code)
    #plt.subplot(1, 3, 1)
    #plt.imshow(img_labels)
    #plt.axis("off")

    #plt.subplot(1, 3, 2)
    #plt.imshow(img_real)
    #plt.axis("off")

    plt.subplot(1, 1, 1)
    plt.imshow(img_code)
    plt.axis("off")

    plt.show()


def plot_reconstructed_img():
    style.use("default")
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_signatures_from_representation_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
            opt.dataset,
            opt.nc,
            opt.n_dis,
            opt.n_conti,
            opt.n_noise,
            opt.epoch,
            opt.outf_suffix),
        "rb"))

    labels = np.array(data["y"]).squeeze()
    real = np.array(data["x"])
    fakes = np.array(data["fake"])

    idx_in_img = np.array(data["origin_indices_in_img"])
    meta = data["meta"]
    dim = meta["dim"]

    sorted_labels = np.array([x for _, x in sorted(zip(idx_in_img, labels))])
    sorted_real_signatures = np.array([x for _, x in sorted(zip(idx_in_img, real))])
    sorted_fake_signatures = np.array([x for _, x in sorted(zip(idx_in_img, fakes))])

    sorted_real_signatures = np.vstack((np.vstack((sorted_real_signatures[:, 59],
                                                   sorted_real_signatures[:, 23])),
                                        sorted_real_signatures[:, 5])).T
    sorted_fake_signatures = np.vstack((np.vstack((sorted_fake_signatures[:, 59],
                                                   sorted_fake_signatures[:, 23])),
                                        sorted_fake_signatures[:, 5])).T

    img_labels = np.reshape(sorted_labels, newshape=[dim[0], dim[1]])
    img_real_signatures = np.reshape(sorted_real_signatures, newshape=[dim[0], dim[1], 3])
    img_fake_signatures = np.reshape(sorted_fake_signatures, newshape=[dim[0], dim[1], 3])

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(img_real_signatures)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_fake_signatures)
    plt.axis("off")

    diff = np.square(img_real_signatures - img_fake_signatures)
    #diff = np.sum(diff, axis=2)  # reduce to one dim
    print(np.min(diff), np.max(diff))
    min_diff = np.min(diff)
    diff -= min_diff
    max_diff = np.max(diff)
    diff/= max_diff
    #plt.subplot(1, 3, 3)
    #plt.imshow(diff)
    #plt.axis("off")

    plt.show()


def plot_manipulated_signatures():
    style.use("default")
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_signatures_from_representation_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
            opt.dataset,
            opt.nc,
            opt.n_dis,
            opt.n_conti,
            opt.n_noise,
            opt.epoch,
            opt.outf_suffix),
        "rb"))

    labels = np.array(data["y"]).squeeze()
    real = np.array(data["x"])
    fakes = np.array(data["fakes_manipulated"])

    idx_in_img = np.array(data["origin_indices_in_img"])
    meta = data["meta"]
    dim = meta["dim"]

    sorted_labels = np.array([x for _, x in sorted(zip(idx_in_img, labels))])
    sorted_real_signatures = np.array([x for _, x in sorted(zip(idx_in_img, real))])
    sorted_fake_signatures = np.array([x for _, x in sorted(zip(idx_in_img, fakes))])

    sorted_real_signatures = np.vstack((np.vstack((sorted_real_signatures[:, 59],
                                                   sorted_real_signatures[:, 23])),
                                        sorted_real_signatures[:, 5])).T
    sorted_fake_signatures = np.vstack((np.vstack((sorted_fake_signatures[:, 59],
                                                   sorted_fake_signatures[:, 23])),
                                        sorted_fake_signatures[:, 5])).T

    img_labels = np.reshape(sorted_labels, newshape=[dim[0], dim[1]])
    img_real_signatures = np.reshape(sorted_real_signatures, newshape=[dim[0], dim[1], 3])
    img_fake_signatures = np.reshape(sorted_fake_signatures, newshape=[dim[0], dim[1], 3])

    plt.figure(figsize=(15,15))
    #plt.subplot(1, 3, 1)
    #plt.imshow(img_labels)
    #plt.axis("off")

    #plt.subplot(1, 3, 2)
    #plt.imshow(img_real_signatures)
    #plt.axis("off")

    plt.subplot(1, 1, 1)
    plt.imshow(img_fake_signatures)
    plt.axis("off")

    plt.show()


def plot_manipulated_signatures_and_representation():
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_code_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
            opt.dataset,
            opt.nc,
            opt.n_dis,
            opt.n_conti,
            opt.n_noise,
            opt.epoch,
            opt.outf_suffix),
        "rb"))

    labels = np.array(data["y"]).squeeze()
    real = np.array(data["x"])
    code = np.array(data["z"]).copy()

    idx_in_img = np.array(data["origin_indices_in_img"])

    meta = data["meta"]
    dim = meta["dim"]

    # tmp = code.copy()
    # tmp[:, 2] = 0.5
    # code[:, 2] = 1
    # code[labels == 1] = tmp[labels == 1]
    # img = np.zeros([dim[0], dim[1], 3])
    sorted_labels = np.array([x for _, x in sorted(zip(idx_in_img, labels))])
    sorted_code = np.array([x for _, x in sorted(zip(idx_in_img, code))])
    sorted_real = np.array([x for _, x in sorted(zip(idx_in_img, real))])

    sorted_real = np.vstack((np.vstack((sorted_real[:, 59],
                                        sorted_real[:, 23])),
                             sorted_real[:, 5])).T

    sorted_code_normed = np.zeros_like(sorted_code, dtype=int)

    color_mode = "rgb"  # "rgb"
    for idx in range(sorted_code.shape[1]):
        tmp = sorted_code[:, idx].copy()
        min_tmp = np.amin(tmp)
        tmp -= min_tmp
        max_tmp = np.amax(tmp)
        if max_tmp != 0:
            tmp /= max_tmp
        if color_mode == "rgb":
            tmp *= 255

        sorted_code_normed[:, idx] = tmp.astype(int)

    if color_mode == "hsv":
        sorted_code_normed = hsv_to_rgb(sorted_code_normed)

    img_labels = np.reshape(sorted_labels, newshape=[dim[0], dim[1]])
    img_code = np.reshape(sorted_code_normed, newshape=[dim[0], dim[1], 3])
    img_real = np.reshape(sorted_real, newshape=[dim[0], dim[1], 3])

    # print(img_code)
    plt.figure(figsize=[32, 15])
    plt.subplot(2, 2, 1)
    plt.imshow(img_labels)
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(2, 2, 2)
    plt.imshow(img_real)
    plt.axis("off")
    plt.title("Real RGB image")

    plt.subplot(2, 2, 3)
    plt.imshow(img_code)
    plt.axis("off")
    plt.title("Learned Representation as RGB image")

    style.use("default")
    data = pickle.load(open(
        "/home/patrick/repositories/hyperspectral_phenotyping_gan/experiments/generated_signatures_from_representation_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
            opt.dataset,
            opt.nc,
            opt.n_dis,
            opt.n_conti,
            opt.n_noise,
            opt.epoch,
            opt.outf_suffix),
        "rb"))

    fakes = np.array(data["fakes_manipulated"])

    idx_in_img = np.array(data["origin_indices_in_img"])
    meta = data["meta"]
    dim = meta["dim"]

    sorted_fake_signatures = np.array([x for _, x in sorted(zip(idx_in_img, fakes))])

    sorted_fake_signatures = np.vstack((np.vstack((sorted_fake_signatures[:, 59],
                                                   sorted_fake_signatures[:, 23])),
                                        sorted_fake_signatures[:, 5])).T

    img_fake_signatures = np.reshape(sorted_fake_signatures, newshape=[dim[0], dim[1], 3])

    plt.subplot(2, 2, 4)
    plt.imshow(img_fake_signatures)
    plt.axis("off")
    plt.title("Manipulation of diseased spots towards healthy")

    plt.show()


if __name__ == '__main__':
    # plot_conti_code_tsne()
    #plot_conti_code_as_img()
    #plot_manipulated_signatures()
    plot_reconstructed_img()
    #plot_manipulated_signatures_and_representation()
