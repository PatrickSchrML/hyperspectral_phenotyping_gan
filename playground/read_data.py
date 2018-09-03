'''
Created on 31.03.2016

@author: mirwaes
'''

import matplotlib.pyplot as plt

from scipy import sparse
import scipy.io as sio
import numpy as np
import spectral as sp
import time

if __name__ == "__main__":
    fname = "cerc15dai175.mat"

    wl = 160
    rfl = 50

    x = sio.loadmat(fname)
    print(x.keys())
    # data dimensionality
    dim = x["dim"].reshape(-1)

    # original counts
    counts = x["counts"]

    # sparse counts after wordification with R=50, maxvalue was set to 1
    counts_sparse = x["counts_sparse"]

    # sparse counts after wordification with R=50 on normalized data by the Euclidean length, maxvalue was set to 0.15
    counts_sparse_norm = x["counts_sparse_norm"]

    # wavelength's for the data
    wavelength = np.array(x["wavelength"][0])
    print(wavelength)

    # rgb image
    plt.subplot(2, 2, 1)
    plt.imshow(x["rgb"])
    plt.axis("off")
    plt.title("RGB")
    # labels for each pixel
    plt.subplot(2, 2, 3)
    plt.axis("off")
    print(x["labels"][:10])

    plt.imshow(x["labels"].reshape(dim[0], dim[1]))
    plt.axis("off")
    plt.title("labels")

    # a visualization of an example discrete signature
    id = 1471#np.random.randint(0, x["counts_sparse"].shape[1])
    example = np.rot90(x["counts_sparse"][:, id].toarray().reshape(wl, rfl))

    plt.subplot(2, 2, 4)
    plt.imshow(example)
    plt.yticks([0, 10, 20, 30, 40, 50], [1, .8, .6, .4, .2, 0])
    plt.xticks(range(0, 160, 30), np.array(wavelength[range(0, 160, 30)], np.int))
    #     plt.colorbar()
    plt.title("discrete")

    plt.subplot(2, 2, 2)

    plt.plot(wavelength, counts[:, id])
    plt.ylim(0, 1)
    plt.xlim(wavelength.min(), wavelength.max())
    plt.xticks(np.array(wavelength[range(0, 160, 30)], np.int))
    plt.title("original")

    print(counts.shape, counts.__class__, counts.dtype, (wavelength.min(), wavelength.max()))
    plt.show()
