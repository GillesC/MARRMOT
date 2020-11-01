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
    ax = plt.gca()

    diff = {
        "URA": [],
        "ULA": []
    }

    for d in dirs:
        # only process 868 measurements
        is_868_b = is_868(d)
        is_ula_b = is_ula(d)

        if not is_868_b:
            # print(f"Skipped {d}")
            continue
        # print(f"processing {d}")
        H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
        # remove faulty antenna 32
        # [snapshots x freq points x BS antennas]
        H = H[:, :, :31]

        path = get_path(d)
        point = get_point(d)
        conf = "ULA" if is_ula(d) else "URA"

        gain = np.abs(H) ** 2

        if not is_cont_meas(d):
            h = np.mean(gain, axis=0)
            n_gain = h[0,:]
            n_min = np.min(n_gain)
            n_max = np.max(n_gain)
            if n_min != 0:
                diff[conf].append(10 * np.log10(n_max) - 10 * np.log10(n_min))

    ula_median = np.nanmedian(diff["ULA"])
    ula_max = np.nanmax(diff["ULA"])

    print(f"Median ULA: {ula_median}dB")
    print(f"Max ULA: {ula_max}dB")

    ura_median = np.nanmedian(diff["URA"])
    ura_max = np.nanmax(diff["URA"])

    print(f"Median URA: {ura_median}dB")
    print(f"Max URA: {ura_max}dB")


