#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/6/19 14:16
"""
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from collections import Counter
from itertools import permutations, combinations
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans, AgglomerativeClustering
from sklearn import preprocessing
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图

from utils.utils import mix_matrix_random, rectangular_to_polar
from span import signal_reconstruction

DATA_1 = r'./data/sig_1.mat'
DATA_2 = r'./data/sig_2.mat'
DATA_3 = r'./data/sig_3.mat'
DATA_4 = r'./data/sig_4.mat'


def get_observe(S, observe_num=2, source_num=3, randomA=True):
    S /= S.std(axis=0)  # Standardize data
    if randomA:
        A = np.array(np.random.uniform(0.2, 1, (observe_num, source_num)))
        A = preprocessing.normalize(A, norm='l2', axis=0)
        FLAG = mix_matrix_random(A, Thresh=0.07)
        while FLAG:
            print("re-random")
            A = np.array(np.random.uniform(0.2, 1, (observe_num, source_num)))
            A = preprocessing.normalize(A, norm='l2', axis=0)
            FLAG = mix_matrix_random(A, Thresh=0.07)
    else:
        A = np.array([[0.4969, 0.1503, 0.7956],
                      [0.8678, 0.9886, 0.6058]])  # Mixing matrixX = np.dot(S, A.T)  # Generate obsZervations
        # A = np.array([[1, .5, 1],
        #               [1, 1, .5]])  # Mixing matrixX = np.dot(S, A.T)  # Generate obsZervations
        # A = np.array([[-0.7618, .9839, -0.1129],
        #               [.6478, .1786, .9929]])
        k_lst = []
        for a in A.T:
            # print(a)
            k = a[0] / (a[1] + 1e-5)
            k_lst.append(k)
        k = np.array(k_lst)
        print("mix matrix k:", k)
    print("-----------------------------------------------")
    print("mix matrix:")
    print(A)

    X = np.dot(S, A.T)  # Generate observations
    # S += 0.05 * np.random.normal(size=S.shape)  # Add noise,

    return X, A


def stft_SSP(X):
    l, n = X.shape
    Z_lst = []

    for i in range(n):
        x = X[:, i]
        f_0, t_0, zxx = scipy.signal.stft(x, fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
        Z_lst.append(zxx)

    Z = np.array(Z_lst).reshape([n, -1]).T
    Z_real = Z.real
    Z_img = Z.imag

    Z_real_norm = preprocessing.normalize(Z_real, norm='l2')
    Z_img_norm = preprocessing.normalize(Z_img, norm='l2')

    # SSP-1
    SSP_1 = Z_real_norm - Z_img_norm
    SSP_1 = np.abs(SSP_1)

    aim = 0.005
    del_SSP_1_idx = []
    for i, data in enumerate(SSP_1):
        if data[0] > aim and data[1] > aim:
            del_SSP_1_idx.append(i)

    # SSP-2
    SSP_2 = np.abs((((Z_real[:, 0] / (Z_img[:, 0] + 1e-5)) - (Z_real[:, 1] / (Z_img[:, 1] + 1e-5))) /
                    ((Z_real[:, 0] / (Z_img[:, 0] + 1e-5)) + (Z_real[:, 1] / (Z_img[:, 1] + 1e-5)))))
    SSP_2 = np.abs(SSP_2)

    aim = 0.05
    del_SSP_idx = []
    for i, data in enumerate(SSP_2):
        if data > aim:
            del_SSP_idx.append(i)
    # del_SSP_idx = []

    threshold = 0.2
    del_real_filter_idx = []
    for i, data in enumerate(Z_real):
        if (np.abs(data[0]) < threshold) and (np.abs(data[1]) < threshold) or data[1] < 0:
            del_real_filter_idx.append(i)

    del_img_filter_idx = []
    for i, data in enumerate(Z_img):
        if (np.abs(data[0]) < threshold) and (np.abs(data[1]) < threshold):
            del_img_filter_idx.append(i)

    del_idx = list(set(del_SSP_idx).union(del_real_filter_idx, del_img_filter_idx, del_SSP_1_idx))
    Z_filted = np.delete(Z, del_idx, axis=0)
    Z_filted = np.r_[Z_filted.real, Z_filted.imag]
    Z_filted_norm = preprocessing.normalize(Z_filted, norm='l2')
    return X, Z_lst, Z_filted, Z_filted_norm


def matrix_estimation(Z, Z_norm, useHough=False):
    global new_Z
    global new_Z_norm
    global label_pred
    clustering = DBSCAN(eps=0.01, min_samples=10).fit(Z_norm)
    label_pred = clustering.labels_
    core = clustering.core_sample_indices_
    # print(core)
    # print(label_pred)

    d2 = Counter(label_pred)
    # print(d2)
    aim_idx = []
    sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_x)
    if len(sorted_x) > 3:
        for x in sorted_x[3:]:
            for i, idx in enumerate(label_pred):
                if x[0] == idx:
                    aim_idx.append(i)
                else:
                    continue
        # print(aim_idx)
        new_Z = np.delete(Z, aim_idx, axis=0)
        new_Z_norm = np.delete(Z_norm, aim_idx, axis=0)
        label_pred = np.delete(label_pred, aim_idx, axis=0)

        d2 = Counter(label_pred)
        sorted_x = sorted(d2.items(), key=lambda x: x[1], reverse=True)
        # print(sorted_x)

    CENTER_1 = []
    CENTER_2 = []
    CENTER_3 = []

    if useHough:
        Rou, the = rectangular_to_polar(Z[:, 0], Z[:, 1])

        for i, idx in enumerate(label_pred):
            if idx == 0:
                CENTER_1.append(the[i])
            elif idx == 1:
                CENTER_2.append(the[i])
            elif idx == 2:
                CENTER_3.append(the[i])
            else:
                continue
        C_1 = math.radians(np.mean(CENTER_1))
        C_2 = math.radians(np.mean(CENTER_2))
        C_3 = math.radians(np.mean(CENTER_3))
        print("-----------------------------------------------")
        print("estimated degrees:", np.degrees(C_1), np.degrees(C_2), np.degrees(C_3))
        print("estimated k:", np.tan(C_1), np.tan(C_2), np.tan(C_3))
        A_ = np.array([[np.cos(C_1), np.cos(C_2), np.cos(C_3)],
                       [np.sin(C_1), np.sin(C_2), np.sin(C_3)]])
        print("estimated A:", (A_))

    else:
        """
        非霍夫变换矩阵估计方法1--常见
        """
        # print("-----------------------------------------------")
        # for i, idx in enumerate(label_pred):
        #     if idx == 0:
        #         CENTER_1.append(new_Z_norm[i])
        #     elif idx == 1:
        #         CENTER_2.append(new_Z_norm[i])
        #     elif idx == 2:
        #         CENTER_3.append(new_Z_norm[i])
        #     else:
        #         continue
        # CENTER_1 = np.array(CENTER_1)
        # CENTER_2 = np.array(CENTER_2)
        # CENTER_3 = np.array(CENTER_3)
        # C_1 = np.round(np.mean(CENTER_1, axis=0), decimals=6)
        # C_2 = np.round(np.mean(CENTER_2, axis=0), decimals=6)
        # C_3 = np.round(np.mean(CENTER_3, axis=0), decimals=6)
        # print(np.array([C_1, C_2, C_3]).T)

        """
        非霍夫变换矩阵估计方法2--PN
        """
        print("-----------------------------------------------")
        for i, idx in enumerate(label_pred):
            if idx == 0:
                CENTER_1.append([new_Z_norm[i, 0], new_Z_norm[i, 1]])
            elif idx == 1:
                CENTER_2.append([new_Z_norm[i, 0], new_Z_norm[i, 1]])
            elif idx == 2:
                CENTER_3.append([new_Z_norm[i, 0], new_Z_norm[i, 1]])
            else:
                continue
        # print(CENTER_1)
        # print(CENTER_2)
        # print(CENTER_3)
        K_1 = np.mean(CENTER_1, axis=0)
        K_2 = np.mean(CENTER_2, axis=0)
        K_3 = np.mean(CENTER_3, axis=0)
        # print(K_1)
        a11 = K_1[0] / np.sqrt(np.power(K_1[0], 2) + np.power(K_1[1], 2))
        a12 = K_2[0] / np.sqrt(np.power(K_2[0], 2) + np.power(K_2[1], 2))
        a13 = K_3[0] / np.sqrt(np.power(K_3[0], 2) + np.power(K_3[1], 2))
        a21 = K_1[1] / np.sqrt(np.power(K_1[0], 2) + np.power(K_1[1], 2))
        a22 = K_2[1] / np.sqrt(np.power(K_2[0], 2) + np.power(K_2[1], 2))
        a23 = K_3[1] / np.sqrt(np.power(K_3[0], 2) + np.power(K_3[1], 2))
        # print(a11)
        A_ = np.array([[a11, a12, a13], [a21, a22, a23]])
        print("estimated A:")
        print((A_))
        """
        非霍夫变换矩阵估计方法3--PCA
        """
        # print("-----------------------------------------------")
        # for i, idx in enumerate(label_pred):
        #     if idx == 0:
        #         CENTER_1.append(new_Z_norm[i])
        #     elif idx == 1:
        #         CENTER_2.append(new_Z_norm[i])
        #     elif idx == 2:
        #         CENTER_3.append(new_Z_norm[i])
        #     else:
        #         continue
        # CENTER_1 = np.array(CENTER_1)
        # CENTER_2 = np.array(CENTER_2)
        # CENTER_3 = np.array(CENTER_3)
        # pca = PCA(n_components=2)
        # newData = pca.fit_transform(CENTER_1.T)
        # print(newData)
        # exit()
        # print(np.array([C_1, C_2, C_3]).T)
    return new_Z, new_Z_norm, label_pred, A_


def visualize(X, zxx, new_Z, new_Z_norm, label_pred):
    plt.figure(figsize=(8, 8))

    circle_1 = plt.Circle((0, 0), 0.2, color='r', fill=False, alpha=0.5, linestyle='--')
    plt.gcf().gca().add_artist(circle_1)

    circle = plt.Circle((0, 0), 1, color='b', fill=True, alpha=0.05, lw=0)
    plt.gcf().gca().add_artist(circle)

    # plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.scatter(zxx[0].real, zxx[1].real, s=.5, alpha=0.5)
    plt.scatter(new_Z[:, 0], new_Z[:, 1], s=1)

    plt.scatter(new_Z_norm[:, 0], new_Z_norm[:, 1], s=5, c=label_pred)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.show()


def pinv(X, A):
    A_1 = np.linalg.pinv(A)
    # print("pinv A:", A_1.shape)
    S = np.dot(X, A_1.T)
    # print("pinv S:", S.shape)
    return S


if __name__ == '__main__':
    s1 = loadmat(DATA_1)
    s2 = loadmat(DATA_2)
    s3 = loadmat(DATA_3)
    s4 = loadmat(DATA_4)

    # S = np.c_[s1['sig_1'][0], s2['sig_2'][0], s3['sig_3'][0], s4['sig_4'][0]]
    S = np.c_[s1['sig_1'][0], s2['sig_2'][0], s3['sig_3'][0]]

    X, A = get_observe(S, observe_num=2, source_num=3)
    X, Z_lst, Z_filted, Z_filted_norm = stft_SSP(X)
    new_Z, new_Z_norm, label_pred, A_ = matrix_estimation(Z_filted, Z_filted_norm)
    visualize(X, Z_lst, new_Z, new_Z_norm, label_pred)

    # print(np.array(Z_lst).shape)
    # exit()
    Smt = signal_reconstruction(np.array(Z_lst), A_)
    plt.pcolormesh(np.abs(np.array(Z_lst)[0, :, :]), shading='gouraud')
    plt.show()
    plt.pcolormesh(np.abs(Smt[0, :, :]), shading='gouraud')
    plt.show()
    plt.pcolormesh(np.abs(Smt[1, :, :]), shading='gouraud')
    plt.show()
    plt.pcolormesh(np.abs(Smt[2, :, :]), shading='gouraud')
    plt.show()
