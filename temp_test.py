#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok827@gmail.com
@Date: 2023/6/30 11:01
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
dt = 0.0005
t = np.arange(0.0, 20.0, dt)
print(t.shape)

s1 = np.sin(2 * np.pi * 100 * t)
s2 = 20 * np.sin(2 * np.pi * 400 * t)

# s2[t <= 10] = s2[12 <= t] = 0
nse = 0.01 * np.random.random(size=len(t))
x = s1 + 0.1*s2 + nse  # the signal


NFFT = 1024  # 窗口段长度
Fs = int(1.0 / dt)  # 频率
fig, (ax1, ax2) = plt.subplots(nrows=2)
ax1.plot(t, x)

Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=900)
plt.show()
