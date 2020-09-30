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
            H =H[:, :, 0:31]
            # average over freq points
            H_avg = np.mean(H, axis=1)
            H_avg = H_avg.transpose() # so it is a MxN matrix
            cov_mat = H_avg@H_avg.conj().transpose()
            M = H.shape[2]
            angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
            pspectrum, psindB, peaks = alg.music(cov_mat, 1, M, angles)
            plt.cla()
            plt.plot(angles, psindB)
            plt.plot(angles[peaks], psindB[peaks], 'x')
            plt.legend(['pseudo spectrum', 'Estimated DoAs'])
            plt.savefig(os.path.join(root_dir, d, "power_spectrum.png"))

            if len(peaks) > 0:
                print(d)






