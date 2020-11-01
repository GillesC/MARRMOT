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

if __name__ == '__main__':
    root_dir = load_root_dir()

    pbar = tqdm(total=6)

    H_a2_ula = np.load(os.path.join(root_dir, "A-2-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a2_ura = np.load(os.path.join(root_dir, "A-2-a-URA-868", "small-channel.npy"))[:, :, :31]

    H_a3_ula = np.load(os.path.join(root_dir, "A-3-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a3_ura = np.load(os.path.join(root_dir, "A-3-a-URA-868", "small-channel.npy"))[:, :, :31]

    H_a9_ula = np.load(os.path.join(root_dir, "A-9-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a9_ura = np.load(os.path.join(root_dir, "A-9-a-URA-868", "small-channel.npy"))[:, :, :31]

    H_a10_ula = np.load(os.path.join(root_dir, "A-10-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a10_ura = np.load(os.path.join(root_dir, "A-10-a-URA-868", "small-channel.npy"))[:, :, :31]
    res = np.zeros(31)

    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1 # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr = compute_corr_coefficient(H_a2_ula[s, 0, start_window:M+start_window], H_a10_ula[s, 0, start_window:M+start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr

            R += R_temp/num_windows
        res[M-1] = R/S
    plt.plot(res, label="A2-A10 ULA")

    pbar.update()



    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1 #(31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr= compute_corr_coefficient(H_a2_ura[s, 0, start_window:M+start_window], H_a10_ura[s, 0, start_window:M+start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr
            R += R_temp/num_windows
        res[M-1] = R/S
    plt.plot(res, label="A2-A10 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a9_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1  # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr = compute_corr_coefficient(H_a9_ura[s, 0, start_window:M + start_window],
                                                   H_a10_ura[s, 0, start_window:M + start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr

            R += R_temp / num_windows
        res[M - 1] = R / S
    plt.plot(res, label="A9-A10 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a9_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1  # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr = compute_corr_coefficient(H_a9_ula[s, 0, start_window:M + start_window],
                                                   H_a10_ula[s, 0, start_window:M + start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr

            R += R_temp / num_windows
        res[M - 1] = R / S
    plt.plot(res, label="A9-A10 ULA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1  # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr = compute_corr_coefficient(H_a2_ura[s, 0, start_window:M + start_window],
                                                   H_a3_ura[s, 0, start_window:M + start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr
            R += R_temp / num_windows
        res[M - 1] = R / S
    plt.plot(res, label="A2-A3 URA")

    pbar.update()

    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = (31 - M) + 1  # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                corr = compute_corr_coefficient(H_a2_ula[s, 0, start_window:M + start_window],
                                                   H_a3_ula[s, 0, start_window:M + start_window])
                if corr is None:
                    num_windows -= 1
                else:
                    R_temp += corr
            R += R_temp / num_windows
        res[M - 1] = R / S
    plt.plot(res, label="A2-A3 ULA")

    pbar.update()




    plt.legend()
    from plotting import LatexifyMatplotlib as lm

    lm.save("corr-coefficient-los-vs-nlos.tex", scale_legend=0.7, show=True, plt=plt)
    #plt.legend()
    #plt.show()
