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

    # wavelength's for the data
    wavelength = np.array(x["wavelength"][0])

    plt.figure(figsize=(32, 16))
    # rgb image included in data
    plt.subplot(1, 2, 1)
    plt.imshow(x["rgb"])
    plt.axis("off")
    plt.title("RGB included in dataset")

    # rgb image from the 3 rgb wavelength
    plt.subplot(1, 2, 2)
    counts_rgb = counts.copy().T
    counts_rgb = np.vstack((np.vstack((counts_rgb[:, 59], counts_rgb[:, 23])), counts_rgb[:, 5])).T
    plt.imshow(np.reshape(counts_rgb, newshape=[dim[0], dim[1], -1]))
    plt.axis("off")
    plt.title("RGB from signatures: \n r: {},\n g: {},\n b: {}".format(round(wavelength[59]),
                                                                       round(wavelength[23]),
                                                                       round(wavelength[5])))
    plt.show()
