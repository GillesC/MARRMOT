import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868
import matplotlib.pyplot as plt

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_corr_coefficient(h1, h2):
    h1_H = np.conjugate(h1.reshape(1, -1))
    h2 = h2.reshape(-1, 1)

    return np.asscalar(np.abs(np.dot(h1_H, h2))**2 / (np.linalg.norm(h1)**2 * np.linalg.norm(h2)**2))


def normalize(h):
    # [snapshots x freq points x BS antennas]
    h_norm = np.zeros_like(h)
    N = h.shape[0]
    F = h.shape[1]
    M = h.shape[2]

    for n in range(N):
        h_norm[n, :, :] = h[n, :, :] / (np.sqrt((1 / (F * M)) * np.sum(np.abs(h[n, :, :]) ** 2)))

    return h_norm


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
        H = normalize(H)
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
    H_ula, H_ura = load_channels(dirs)

    antenna_idx = list(range(31))

    num_simulations = 100000

    res = {
        "ULA": np.zeros(shape=(31, num_simulations)),
        "URA": np.zeros(shape=(31, num_simulations)),
        "iid-sim": np.zeros(shape=(31, num_simulations)),
        "iid": np.zeros(shape=(31, num_simulations)),
        # "iid_half": np.zeros(shape=(31, num_simulations)),
        # "iid_double": np.zeros(shape=(31, num_simulations)),
        # "iid_sim": np.zeros(shape=(31, num_simulations)),
    }
    pbar = tqdm(total=num_simulations*31*4)
    for sim in range(0, num_simulations):
        for num_antennas in range(1, 32):
            for conf, H_conf in zip(["ULA", "URA", "iid-sim", "iid"], [H_ula, H_ura, None, None]):
                if "iid" in conf:
                    # if conf == "iid_half":
                    #     var = np.sqrt(0.5)
                    # elif conf == "iid_double":
                    #     var = np.sqrt(2)
                    # else:
                    if "iid-sim" == conf:
                        var = 1
                        h_1 = np.sqrt(1 / (2 * var)) * (
                                np.random.normal(0, var, num_antennas) + 1j * np.random.normal(0, var, num_antennas))
                        h_2 = np.sqrt(1 / (2 * var)) * (
                                np.random.normal(0, var, num_antennas) + 1j * np.random.normal(0, var, num_antennas))
                else:
                    num_pos = len(H_conf)
                    pos_1 = random.randint(0, num_pos - 1)
                    pos_2 = random.randint(0, num_pos - 1)

                    while pos_1 == pos_2:
                        pos_2 = random.randint(0, num_pos - 1)

                    # only two frequencies
                    freq = random.randint(0, 1)

                    random.shuffle(antenna_idx)
                    random_antennas = antenna_idx[:num_antennas]

                    h_1 = H_conf[pos_1]
                    h_2 = H_conf[pos_2]

                    time_1 = random.randint(0, h_1.shape[0] - 1)
                    time_2 = random.randint(0, h_2.shape[0] - 1)

                    h_1 = h_1[time_1, freq, random_antennas]
                    h_2 = h_2[time_2, freq, random_antennas]

                c = compute_corr_coefficient(h_1, h_2)
                if "iid" == conf:
                    c = 1/num_antennas
                res[conf][num_antennas - 1][sim] = c
                pbar.update()

    plt.cla()
    fig, ax = plt.subplots()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(ax, width="40%", height=1.5, loc=9)
    for conf, values in res.items():
        # if conf == "iid":
        #     avg_c = np.sqrt(1 / np.arange(1, 32))
        # else:
        avg_c = np.nanmean(values, axis=1)
        ax.plot(np.arange(1, 32), avg_c, marker='o', label=conf, linewidth=2, markersize=3,
                alpha=0.5)
        axins.plot(avg_c, marker='o', label=conf, linewidth=2, markersize=3,
                   alpha=0.5)

        # plt.plot(avg_c, label=conf, linestyle=linestyle)

    x1, x2, y1, y2 = 4, 8, 0.3, 0.5  # specify the limits
    axins.set_xlim(x1, x2)  # apply the x-limits
    axins.set_ylim(y1, y2)  # apply the y-limits
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.legend()

    from plotting import LatexifyMatplotlib as lm

    lm.save("corr-coefficient.tex", scale_legend=0.7, show=True, plt=plt)
