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
        H = np.mean(H, axis=1)

        if not is_cont_meas(d):
            path = get_path(d)
            point = get_point(d)
            conf = "ULA" if is_ula(d) else "URA"

            if path == "A":
                if point in [2, 10]:
                    gain = np.abs(H) ** 2

                    gain[gain==0] = 0.000000000000000001

                    std = np.std(10 * np.log10(gain),axis=0)
                    mean = np.mean(10 * np.log10(gain), axis=0)

                    color = next(ax._get_lines.prop_cycler)['color']
                    plt.plot(np.arange(1, 32), mean, label=f"{conf}-{point}-{np.std(mean):0.2f} dB std", linewidth=1,
                             color=color)

                    # plt.plot(np.arange(1, 32), mean - np.abs(std), linewidth=1, alpha=0.2, color=color)
                    # plt.plot(np.arange(1, 32), mean + np.abs(std), linewidth=1, alpha=0.2, color=color)

                    # plt.fill_between(np.arange(1, 32), mean - np.abs(std), mean + np.abs(std), alpha=0.1, color=color)
                    print(f"{conf}-{point}-{np.std(mean):0.2f} dB std")



    # plt.legend()
    plt.savefig(f"avg_gain.png", transparent=True)
