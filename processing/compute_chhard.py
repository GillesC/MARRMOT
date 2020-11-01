import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def chhard(h):
    power_h = np.linalg.norm(h)
    return np.var(power_h / np.mean(power_h))


def load_channels(dirs):
    """
    Load all the 868 channels
    If it is a continious measurement we split the channel snapshots per 100 snapshots
    All locations are stored in an array and returned per antenna conf ULA/URA
    """
    H_ula = []
    H_ura = []

    for d in dirs:
        # only process 868 measurements
        is_868_b = is_868(d)
        is_ula_b = is_ula(d)

        if not is_868_b:
            print(f"Skipped {d}")
            continue
        print(f"processing {d}")
        H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
        # remove faulty antenna 32
        H = H[:, :, :-1]
        if is_cont_meas(d):
            # too much samples, split in 1 sec snapshots, i.e. 100 snapshots
            H_splitted = np.array_split(H, H.shape[0] // 100)
            for H_split in H_splitted:
                if is_ula_b:
                    H_ula.append(H_split)
                else:
                    H_ura.append(H_split)
        else:
            if is_ula_b:
                H_ula.append(H)
            else:
                H_ura.append(H)

    return H_ula, H_ura


if __name__ == '__main__':

    antenna_idx = np.arange(0, 31)

    num_simulations = 100

    H_ula, H_ura = load_channels(dirs)

    # preallocate the results
    res = {
        "ULA": np.zeros(shape=(len(H_ula), len(antenna_idx), num_simulations)),
        "URA": np.zeros(shape=(len(H_ura), len(antenna_idx), num_simulations)),
    }

    pbar = tqdm(
        total=(len(antenna_idx) * num_simulations))

    for sim in range(0, num_simulations):
        for M in antenna_idx + 1:
            for conf, H_conf in zip(["ULA", "URA"], [H_ula, H_ura]):
                for p in range(len(H_conf)):
                    h = H_conf[p]
                    random.shuffle(antenna_idx)
                    random_antennas = antenna_idx[:M]
                    h = h[:, :, random_antennas]
                    res[conf][p][M - 1][sim] = chhard(h)
            pbar.update()

    fig, ax = plt.subplots()
    # axins = inset_axes(ax, width="40%", height=1.5, loc=9)
    for conf, p_values in res.items():
        # get the average over all simulations
        p_values = np.nanmean(p_values, axis=(0, 2))
        linestyle = "-"
        if conf == "ULA":
            linestyle = "--"
        ax.plot(np.arange(1, 32), p_values, label=conf, linewidth=1)

    plt.legend(ncol=4)
    plt.show()
