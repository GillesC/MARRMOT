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
    res = []

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
        H = H[:, 0, :31]

        if (not is_cont_meas(d)) and (is_ula(d)):
            H = np.nanmean(H, axis=0)
            # H = np.mean(H, axis=0)
            H_norm = H / np.linalg.norm(H)
            res.append(H_norm)

    H = 20 * np.log10(np.abs(res))
    plt.cla()
    plt.clf()
    plt.imshow(H, cmap='Spectral', aspect='auto')
    plt.gca().invert_xaxis()
    plt.axis('off')
    #plt.colorbar()
    # plt.show()
    plt.savefig(f"waterfall_gain.png", transparent=True, bbox_inches='tight')

    # plt.legend()
    # plt.savefig(f"avg_gain.png", transparent=True)

    from plotting import LatexifyMatplotlib as lm

    lm.save("waterfall_gain.tex", scale_legend=0.7, show=True, plt=plt)
