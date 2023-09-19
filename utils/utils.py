#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Author: JetKwok
@HomePage: https://FMVPJet.github.io/
@E-mail: JetKwok817@gmail.com
@Date: 2023/6/19 14:17
"""

import numpy as np
from itertools import permutations, combinations

"""
- utils for mix matrix estimation
- mixMatrixRandom: check if the mix matrix is random
- Rectangular_to_Polar: convert rectangular coordinates to polar coordinates
"""


def mix_matrix_random(A, Thresh=0.07):
    FLAG = False
    k_lst = []
    for a in A.T:
        # print(a)
        k = a[0] / (a[1] + 1e-5)
        k_lst.append(k)
    k = np.array(k_lst)
    print("mix matrix k:", k)
    comb = combinations(k, 2)
    # print([i[0] - i[1] for i in comb])
    for i in list(comb):
        if np.abs(i[0] - i[1]) < Thresh:
            # print("need re-random")
            # print(np.abs(i[0] - i[1]))
            FLAG = True
            break
    return FLAG


def rectangular_to_polar(x, y):  # 直角坐标转极坐标，输出的thata为角度值
    r = np.sqrt(np.square(x) + np.square(y))
    theta = np.degrees(np.arctan(y / x))
    return r, theta


"""
- utils for signal reconstruction
- STFTMaskFunc: mask function for STFT
"""

