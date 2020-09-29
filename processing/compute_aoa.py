import os
import random

import numpy as np

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868
import matplotlib.pyplot as plt

import aoa_algorithms as alg

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


if __name__ == '__main__':
    for d in dirs:
        if is_868(d) and not is_cont_meas(d) and is_ula(d):
            H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
            # average over all snapshots
            H_avg = np.mean(H, axis=0)
            M = H.shape[2]
            angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            pspectrum, psindB = alg.music(np.cov(H_avg), 1, M, angles)
            plt.plot(angles, psindB)
            plt.save(os.path.join(root_dir, d, "power_spectrum.png"))






