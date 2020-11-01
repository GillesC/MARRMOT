import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path, get_point
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = load_root_dir()
    subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

    pbar = tqdm(total=len(dirs))

    URA = "URA"
    ULA = "ULA"

    for d in dirs:
        # only process 868 measurements
        is_868_b = is_868(d)
        is_ula_b = is_ula(d)

        # if is_cont_meas(d):
        # print(f"processing {d}")
        H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
        # remove faulty antenna 32
        # [snapshots x freq points x BS antennas]
        H = H[:, :, :-1]
        H = np.nanmean(H, axis=1)

        conf = ULA if is_ula(d) else URA

        _min = np.min(H[np.nonzero(H)])
        H[H == 0] = _min

        H = 20 * np.log10(np.abs(H))
        plt.cla()
        plt.clf()
        plt.imshow(H, cmap='viridis', aspect='auto')
        plt.gca().invert_xaxis()
        plt.colorbar()
        plt.title(d)
        plt.savefig(os.path.join(root_dir, d, "waterfall-gain.png"))
        pbar.update()
