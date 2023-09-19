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

DATA_1 = r'./data/sig_1.mat'
DATA_2 = r'./data/sig_2.mat'
DATA_3 = r'./data/sig_3.mat'

A = np.array([[0.4969, 0.1503, 0.7956],
              [0.8678, 0.9886, 0.6058]])

A12 = A[:, 0:2]
A13 = np.c_[A[:, 0], A[:, 2]]
# print(A12)
# print(A13)
A12_inv = (1 / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
A13_inv = (1 / (A[0, 0] * A[1, 2] - A[0, 2] * A[1, 0])) * np.array([[A[1, 2], -A[0, 2]], [-A[1, 0], A[0, 0]]])
# print(A12_inv)
# print(A13_inv)


# source signal S
s1 = loadmat(DATA_1)
s2 = loadmat(DATA_2)
s3 = loadmat(DATA_3)

S = np.c_[s1['sig_1'][0], s2['sig_2'][0], s3['sig_3'][0]]
_, _, St_0 = scipy.signal.stft(S[:, 0], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
_, _, St_1 = scipy.signal.stft(S[:, 1], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
_, _, St_2 = scipy.signal.stft(S[:, 2], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
St = np.array([St_0, St_1, St_2])
print("St-->", St.shape)

# observations X
X = np.dot(S, A.T)

# stft observations F
_, _, Ft_0 = scipy.signal.stft(X[:, 0], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
_, _, Ft_1 = scipy.signal.stft(X[:, 1], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)

F12_lst = []
F13_lst = []
for i in range(len(Ft_0[:, 0])):
    Ft = np.array([Ft_0[i], Ft_1[i]])
    # print(zxx.shape)
    F12 = np.dot(A12_inv, Ft).T
    # print(F12.shape)
    F13 = np.dot(A13_inv, Ft).T
    F12_lst.append(F12)
    F13_lst.append(F13)
    # print(F12.shape)

F12 = np.array(F12_lst).transpose(2, 0, 1)
F13 = np.array(F13_lst).transpose(2, 0, 1)
print("F12-->", F12.shape)

_, S_hat_0 = scipy.signal.istft(F12[0, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
_, S_hat_1 = scipy.signal.istft(F12[1, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
_, S_hat_2 = scipy.signal.istft(F13[1, :, :], fs=25000, window='hann', nfft=512, nperseg=256, noverlap=128)
S_hat_0 = np.expand_dims(S_hat_0, axis=1)
S_hat_1 = np.expand_dims(S_hat_1, axis=1)
S_hat_2 = np.expand_dims(S_hat_2, axis=1)
print(S_hat_0.shape)
# print(xrec.shape)

# print(F12_[:10, 0])
# print(F13_[:10, 0])

# exit()
plt.plot(S_hat_0[10000:10200, 0], label='source')
plt.plot(S_hat_2[10000:10200, 0], label='reconstruction')
plt.title('Signal1 Plot')
plt.legend(loc='best')
plt.show()
