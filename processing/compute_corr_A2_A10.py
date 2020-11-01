import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path, get_point
import matplotlib.pyplot as plt

def compute_corr_coefficient(h1, h2):
    h1_H = np.conjugate(h1.reshape(1, -1))
    h2 = h2.reshape(-1, 1)

    return np.asscalar(np.abs(np.dot(h1_H, h2))**2 / (np.linalg.norm(h1)**2 * np.linalg.norm(h2)**2))

if __name__ == '__main__':
    root_dir = load_root_dir()

    H_a2_ula = np.load(os.path.join(root_dir, "A-2-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a2_ura = np.load(os.path.join(root_dir, "A-2-a-URA-868", "small-channel.npy"))[:, :, :31]

    H_a10_ula = np.load(os.path.join(root_dir, "A-10-a-ULA-868", "small-channel.npy"))[:, :, :31]
    H_a10_ura = np.load(os.path.join(root_dir, "A-10-a-URA-868", "small-channel.npy"))[:, :, :31]
    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ula.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = 1 # (31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                R_temp += compute_corr_coefficient(H_a2_ula[s, 0, start_window:M+start_window], H_a10_ula[s, 0, start_window:M+start_window])

            R += R_temp/num_windows
        res[M-1] = R/S
    plt.plot(res, label="ULA")


    res = np.zeros(31)
    for M in np.arange(1, 32):
        R = 0
        S = H_a2_ura.shape[0]
        for s in range(S):
            # average over different subarrays with M antennas
            num_windows = 1 #(31 - M) + 1
            R_temp = 0
            for start_window in range(num_windows):
                R_temp += compute_corr_coefficient(H_a2_ura[s, 0, start_window:M+start_window], H_a10_ura[s, 0, start_window:M+start_window])

            R += R_temp/num_windows
        res[M-1] = R/S
    plt.plot(res, label="URA")
    plt.axvline(x=8, linestyle="--", color="black", alpha=0.1)
    plt.show()
    #plt.legend()
    #plt.show()
