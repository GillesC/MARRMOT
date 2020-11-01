import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path, get_point
import matplotlib.pyplot as plt

# A2 vs A3 LoS
# A9 vs A10 NLoS
# A2 vs A9 LoS vs NLoS

def compute_corr_coefficient(h1, h2):
    h1_H = np.conjugate(h1.reshape(1, -1))
    h2 = h2.reshape(-1, 1)

    if np.linalg.norm(h1) * np.linalg.norm(h2) == 0:
        return None

    return np.abs(np.dot(h1_H, h2))**2 / (np.linalg.norm(h1)**2 * np.linalg.norm(h2)**2)

def normalize(h):
    # [snapshots x freq points x BS antennas]
    h_norm = np.zeros_like(h)

    # if len(h.shape) == 1:
    #     # ok we got a M-array
    #     return h / np.linalg.norm(h)

    N = h.shape[0]
    F = h.shape[1]
    M = h.shape[2]

    h_norm = (h / np.linalg.norm(h)) * N * F * M
    return h_norm

if __name__ == '__main__':
    root_dir = load_root_dir()

    pbar = tqdm(total=6)

    H_los1_ula = normalize(np.load(os.path.join(root_dir, "A-2-a-ULA-868", "small-channel.npy"))[:, :, :31])
    H_los1_ura = normalize(np.load(os.path.join(root_dir, "A-2-a-URA-868", "small-channel.npy"))[:, :, :31])

    H_a3_ula = normalize(np.load(os.path.join(root_dir, "A-3-a-ULA-868", "small-channel.npy"))[:, :, :31])
    H_a3_ura = normalize(np.load(os.path.join(root_dir, "A-3-a-URA-868", "small-channel.npy"))[:, :, :31])

    H_a9_ula = normalize(np.load(os.path.join(root_dir, "A-4-a-ULA-868", "small-channel.npy"))[:, :, :31])
    H_a9_ura = normalize(np.load(os.path.join(root_dir, "A-4-a-URA-868", "small-channel.npy"))[:, :, :31])

    H_a10_ula = normalize(np.load(os.path.join(root_dir, "A-5-a-ULA-868", "small-channel.npy"))[:, :, :31])
    H_a10_ura = normalize(np.load(os.path.join(root_dir, "A-5-a-URA-868", "small-channel.npy"))[:, :, :31])
    res = np.zeros(31)

    for M in np.arange(1, 32):
        R = 0
        S = H_los1_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_los1_ula[s, 0, :M],
                                            H_a10_ula[s, 0, :M])
            if corr is None:
                S -= 1
            else:
                R += corr
        res[M-1] = R/S
    #plt.plot(res, label="LoS1-NLoS2 ULA")

    pbar.update()



    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_los1_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_los1_ura[s, 0, :M],
                                            H_a10_ura[s, 0, :M])
            R += corr
        res[M-1] = R/S
    #plt.plot(res, label="LoS1-NLoS2 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a9_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_a9_ura[s, 0, :M],
                                            H_a10_ura[s, 0, :M])
            R += corr
        res[M-1] = R/S
    plt.plot(res, label="NLoS1-NLoS2 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a9_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_a9_ula[s, 0, :M],
                                            H_a10_ula[s, 0, :M])
            if corr is None:
                S -= 1
            else:
                R += corr
        res[M-1] = R/S
    plt.plot(res, label="NLoS1-NLoS2 ULA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_los1_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_los1_ura[s, 0, :M],
                                            H_a3_ura[s, 0, :M])
            R += corr
        res[M-1] = R/S
    plt.plot(res, label="LoS1-LoS2 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_los1_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            corr = compute_corr_coefficient(H_los1_ula[s, 0, :M],
                                            H_a3_ula[s, 0, :M])
            R += corr
        res[M-1] = R/S
    plt.plot(res, label="LoS1-LoS2 ULA")

    pbar.update()




    plt.legend()
    from plotting import LatexifyMatplotlib as lm

    lm.save("corr-coefficient-los-vs-nlos-no-windows.tex", scale_legend=0.7, show=True, plt=plt)
    # plt.show()
