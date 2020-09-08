import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula
import random

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_corr_coefficient():
    pass


if __name__ == '__main__':
    H_ula = []
    H_ura = []

    for d in dirs:
        # [snapshots x freq points x BS antennas]
        path = os.path.join(root_dir, d, "small-channel.npy")
        if not is_cont_meas(d):
            H = np.load(input)
            if is_ula(d):
                H_ula.append(H)
            else:
                H_ura.append(H)

    antenna_idx = range(31)
    for num_antennas in range(1, 32):
        for H_conf in [H_ula, H_ura]:
            num_pos = len(H_conf)
            pos_1 = random.randint(0, num_pos)
            pos_2 = random.randint(0, num_pos)

            # only two frequencies
            freq = random.randint(0, 1)

            random.shuffle(antenna_idx)
            random_antennas = antenna_idx[:num_antennas]

            # we have 1000 samples per meas
            time_1 = random.randint(0, 1000)
            time_2 = random.randint(0, 1000)

            h_1 = H_conf[pos_1][time_1, freq, random_antennas]
            h_2 = H_conf[pos_2][time_2, freq, random_antennas]
