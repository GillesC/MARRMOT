import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def _plot(d):
    path = pjoin(root_dir, d)
    input = pjoin(path, "small-channel.npy")
    if os.path.isfile(input):
        H_norm = np.load(input)
        # [snapshots x freq points x BS antennas x users]

        # remove last one bc faulty
        gain_50 = np.abs(H_norm[:, 0, 0:31])
        gain_51 = np.abs(H_norm[:, 1, 0:31])

        # if gain is 0 take the value of the other carrier freq.
        gain_50[gain_50 == 0] = gain_51[gain_50 == 0]
        gain_51[gain_51 == 0] = gain_50[gain_51 == 0]

        z = (20 * np.log10(gain_50) + 20 * np.log10(gain_51)) / 2

        fig, ax = plt.subplots()

        for snapshot in z:
            ax.plot(snapshot, color="blue", alpha=0.001)

        plt.savefig(pjoin(path, "gain-per-snapshot.pdf"))


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(_plot, d)
            future.add_done_callback(lambda p: pbar.update())
