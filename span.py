#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/6/20 17:02
"""

from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import scipy
import time

from scipy import interpolate

np.set_printoptions(threshold=np.inf)

DATA_1 = r'./data/sig_1.mat'
# DATA_2 = r'./data/sig_2.mat'
DATA_2 = r'./data/sig_3.mat'
DATA_3 = r'./data/sig_4.mat'

A = np.array([[0.4969, 0.1503, 0.7956],
              [0.8678, 0.9886, 0.6058]])


def stft_mask_func(Xt, EPS=0.0001):
    Xt = np.abs(Xt)
    Bt_lst = []
    for i in range(Xt.shape[0]):
        Bt = (Xt[i, :, :] - np.min(Xt[i, :, :])) / (np.max(Xt[i, :, :]) - np.min(Xt[i, :, :]))
        mu = Bt.sum() / (Xt.shape[1] * Xt.shape[2])
        eps = 1
        while eps > EPS:
            G1 = Bt.copy()
            G0 = Bt.copy()

            G1[G1 < mu] = 0
            N1 = np.count_nonzero(G1)
            mu1 = np.sum(G1) / N1

            G0[G0 >= mu] = 0
            N0 = np.count_nonzero(G0)
            mu0 = np.sum(G1) / N0

            mu_ = (mu1 + mu0) / 2
            eps = np.abs(mu_ - mu)
            mu = mu_
            del G1, G0

        Bt[Bt < mu] = 0
        Bt[Bt >= mu] = 1
        Bt_lst.append(Bt)
    return np.array(Bt_lst)


def Ar_func(A):
    return np.dot(np.dot(A, np.linalg.inv(np.dot(A.T, A))), A.T)


def D_func(Xt, Pr):
    return np.linalg.norm((Xt - np.dot(Pr, Xt)), ord=2, keepdims=True) / np.linalg.norm(Xt, ord=2, keepdims=True)


def Smt_func(Am, Xt):
    return np.dot(np.dot(np.linalg.inv(np.dot(Am.T, Am)), Am.T), Xt)


def signal_reconstruction(Xt, A):
    t = time.time()
    Smt = np.zeros((3, Xt.shape[1], Xt.shape[2]), dtype=complex)
    Bt = stft_mask_func(Xt)
    MaskedXt = Bt * Xt

    for i in range(Xt.shape[1]):
        for j in range(Xt.shape[2]):
            # print(Xt[:, i, j])
            if np.abs(MaskedXt[0, i, j]) == 0 and np.abs(MaskedXt[1, i, j]) == 0:
                continue
            a_1 = np.expand_dims(A[:, 0], axis=1)
            D_a1 = D_func(Xt[:, i, j], Ar_func(a_1))

            a_2 = np.expand_dims(A[:, 1], axis=1)
            D_a2 = D_func(Xt[:, i, j], Ar_func(a_2))

            a_3 = np.expand_dims(A[:, 2], axis=1)
            D_a3 = D_func(Xt[:, i, j], Ar_func(a_3))
            D = [D_a1, D_a2, D_a3]
            Am = [a_1, a_2, a_3]

            D_min = min(D)

            if D_min < 0.1:
                if D.index(D_min) == 0:
                    Smt[0, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])
                elif D.index(D_min) == 1:
                    Smt[1, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])
                elif D.index(D_min) == 2:
                    Smt[2, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])
            else:
                # continue
                a_12 = A[:, 0:2]
                D_a12 = D_func(Xt[:, i, j], Ar_func(a_12))

                a_13 = np.c_[A[:, 0], A[:, 2]]
                D_a13 = D_func(Xt[:, i, j], Ar_func(a_13))

                a_23 = A[:, 1:3]
                D_a23 = D_func(Xt[:, i, j], Ar_func(a_23))

                D = [D_a12, D_a13, D_a23]
                Am = [a_12, a_13, a_23]

                D_min = min(D)
                if D_min < 0.1:
                    # print(D.index(D_min))
                    # print(SmtFunc(Am[D.index(D_min)], Xt[:, i, j]))
                    # print(SmtFunc(Am[D.index(D_min)], Xt[:, i, j]).shape)
                    if D.index(D_min) == 0:
                        Smt[0, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[0]
                        Smt[1, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[1]
                    elif D.index(D_min) == 1:
                        Smt[0, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[0]
                        Smt[2, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[1]
                    elif D.index(D_min) == 2:
                        Smt[1, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[0]
                        Smt[2, i, j] = Smt_func(Am[D.index(D_min)], Xt[:, i, j])[1]

                # else:
                #     Am = [A[:, 0:3]]
                #     D = DFunc(Xt[:, i, j], ArFunc(Am[0]))
                #     if D < 0.1:
                #         print("33333333")
                #         Smt[:, i, j] = SmtFunc(Am[0], Xt[:, i, j])
    #
    # Bt = STFTMaSkFunc(Smt)
    # Smt = Bt * Smt
    print("Smt-->", Smt.shape)
    print("time-->", time.time() - t)
    return Smt


if __name__ == '__main__':

    # source signal S
    s1 = loadmat(DATA_1)
    s2 = loadmat(DATA_2)
    s3 = loadmat(DATA_3)

    S = np.c_[s1['sig_1'][0], s2['sig_3'][0], s3['sig_4'][0]]
    _, _, St_0 = scipy.signal.stft(S[:, 0], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    _, _, St_1 = scipy.signal.stft(S[:, 1], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    _, _, St_2 = scipy.signal.stft(S[:, 2], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    St = np.array([St_0, St_1, St_2])
    print("St-->", St.shape)

    # observations X
    X = np.dot(S, A.T)

    # stft observations F
    _, _, Xt_0 = scipy.signal.stft(X[:, 0], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    _, _, Xt_1 = scipy.signal.stft(X[:, 1], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)

    Xt = np.array([Xt_0, Xt_1])
    print("Xt-->", Xt.shape)

    Smt = signal_reconstruction(Xt, A)
    _, S_hat_0 = scipy.signal.istft(Smt[0, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    _, S_hat_1 = scipy.signal.istft(Smt[1, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    _, S_hat_2 = scipy.signal.istft(Smt[2, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
    S_hat_0 = np.expand_dims(S_hat_0, axis=1)
    S_hat_1 = np.expand_dims(S_hat_1, axis=1)
    S_hat_2 = np.expand_dims(S_hat_2, axis=1)
    print("S_hat_0-->", S_hat_0.shape)

    # plt.pcolormesh(np.abs(St_1), shading='gouraud')
    # plt.show()

    # print(np.abs(St_2[90:100, :]).T)
    # print(np.abs(Smt[2, 90:100, :]).T)
    # plt.pcolormesh(np.abs(Smt[0, :, :]), shading='gouraud')
    # plt.show()
    #
    # exit()
    print(np.abs(Smt[1, :, :]).shape)
    print(St_1 - np.abs(Smt[1, :, :]))

    show_with_stft = True
    if show_with_stft:
        plt.pcolormesh(np.abs(St_1), shading='gouraud', cmap='gray')
        # plt.pcolormesh(np.abs(Xt_0), shading='gouraud')
        plt.show()
        # plt.pcolormesh(np.abs(Smt[0, :, :]), shading='gouraud')
        # plt.show()
        plt.pcolormesh(np.abs(Smt[1, :, :]), shading='gouraud', cmap='gray')
        plt.show()
        # plt.pcolormesh(np.abs(Smt[2, :, :]), shading='gouraud')
        # plt.show()

    else:
        m = 0
        n = 500
        plt.subplot(3, 1, 1)
        plt.plot(S[m:n, 0], c='r')
        plt.plot(S_hat_0[m:n, 0], c='b')
        plt.title('Signal Plot')

        plt.subplot(3, 1, 2)
        plt.plot(S[m:n, 1], c='r')
        plt.plot(S_hat_1[m:n, 0], c='b')

        plt.subplot(3, 1, 3)
        plt.plot(S[m:n, 2], c='r')
        plt.plot(S_hat_2[m:n, 0], c='b')
        plt.show()

        plt.subplot(3, 1, 1)
        plt.plot(S_hat_0[m:n, 0], c='b')
        plt.plot(S[m:n, 0], c='r')
        plt.title('Signal Plot')

        plt.subplot(3, 1, 2)
        plt.plot(S_hat_1[m:n, 0], c='b')
        plt.plot(S[m:n, 1], c='r')

        plt.subplot(3, 1, 3)
        plt.plot(S_hat_2[m:n, 0], c='b')
        plt.plot(S[m:n, 2], c='r')

        plt.show()
