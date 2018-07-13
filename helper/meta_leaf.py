import scipy.io
import os

import numpy as np


def normalize(data):
    #data -= np.min(data)
    data /= np.max(data)
    return data

file_name = 'cerc15dai175'
file_path = '/home/patrick/repositories/datasets/mirwaes/' + file_name + '.mat'

input_mat = scipy.io.loadmat(os.path.join(file_path))['counts']
input_mat = input_mat.T
# input_mat = np.reshape(input_mat, newshape=[123, 147, input_mat.shape[1]])
input_mat = input_mat.astype(float)

data = input_mat
input_mat_gt = scipy.io.loadmat(os.path.join(file_path))['labels'].T.flatten() - 1

data_origin = data.copy()
targets_origin = input_mat_gt.copy()

data = normalize(data)  # normalize data to be between [0,1]

code_mean = np.mean(data[:, 75] - data[:, 30])
code_max_g = np.amax(data[:, 75] - data[:, 30])
code_min_g = np.amin(data[:, 75] - data[:, 30])
ab = np.maximum(np.abs(code_max_g), np.abs(code_min_g))
code_max = np.amax((data[:, 75] - data[:, 30] - code_mean) / (ab))
code_min = np.amin((data[:, 75] - data[:, 30] - code_mean) / (ab))
code_mean_a = np.mean((data[:, 75] - data[:, 30] - code_mean) / (ab))

print(code_mean)
print(ab)
print(code_mean_a)
print(code_max)
print(code_min)
"""
class0_x, _ = data[input_mat_gt == 0], input_mat_gt[input_mat_gt == 0]  # 14422
class1_x, _ = data[input_mat_gt == 1], input_mat_gt[input_mat_gt == 1]  # 2230
class2_x, _ = data[input_mat_gt == 2], input_mat_gt[input_mat_gt == 2]  # 1429

code0_mean_class0 = np.mean(class0_x[:, 30] - class0_x[:, 75])
code0_max_class0_g = np.amax(class0_x[:, 30] - class0_x[:, 75])
code0_max_class0 = np.amax((class0_x[:, 30] - class0_x[:, 75] - code0_mean_class0) / code0_max_class0_g)
code0_min_class0 = np.amin((class0_x[:, 30] - class0_x[:, 75] - code0_mean_class0) / code0_max_class0_g)

print(code0_mean_class0)
print(code0_max_class0)
print(code0_min_class0)
print("---")"""

"""
code0_mean_class1 = np.mean(class1_x[:, 30] - class1_x[:, 75])
code0_max_class1_g = np.amax(class1_x[:, 30] - class1_x[:, 75])
code0_max_class1 = np.amax(class1_x[:, 30] - class1_x[:, 75])
code0_min_class1 = np.amin(class1_x[:, 30] - class1_x[:, 75])

print(code0_mean_class1)
print(code0_max_class1)
print(code0_min_class1)
print("---")
code0_mean_class2 = np.mean(class2_x[:, 30] - class2_x[:, 75])
code0_max_class2 = np.amax(class2_x[:, 30] - class2_x[:, 75])
code0_min_class2 = np.amin(class2_x[:, 30] - class2_x[:, 75])

print(code0_mean_class2)
print(code0_max_class2)
print(code0_min_class2)
"""