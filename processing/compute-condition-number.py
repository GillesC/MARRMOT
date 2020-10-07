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


def normalize(h):
    # [snapshots x freq points x BS antennas]
    h_norm = np.zeros_like(h)

    if len(h.shape) == 1:
        # ok we got a M-array
        return h / np.linalg.norm(h)

    N = h.shape[0]

    for n in range(N):
        # h_norm[n, :, :] = h[n, :, :] / ((1 / (F * M)) * np.sqrt(np.sum(np.abs(h[n, :, :]) ** 2)))
        h_norm[n, :, :] = h[n, :, :] / np.linalg.norm(h[n, :, :])
    return h_norm


def compute_condition_num(H_list, M, K):
    H = np.zeros(shape=(M, K), dtype=complex)
    for k in range(K):
        H[:, k] = H_list[k]

    # assert H.shape == (M, K)

    H_H = np.conjugate(H).T

    # assert H_H.shape == (K, M)

    H_corr = np.dot(H_H, H)

    # assert H_corr.shape == (K, K)

    # u, s, vh = np.linalg.svd(H_corr)

    eigvals = np.abs(np.linalg.eigvals(H_corr))
    max_eigval = np.max(eigvals)
    min_eigval = np.min(eigvals)

    if min_eigval == 0:
        return 0

    return min_eigval / max_eigval

    # return np.min(s) / np.max(s)


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

    antenna_idx = np.arange(0, 31)

    num_simulations = 10000
    num_nodes = [2, 5, 10]

    H_ula, H_ura = load_channels(dirs)

    # preallocate the results
    res = {
        "ULA": np.zeros(shape=(len(num_nodes), len(antenna_idx), num_simulations)),
        "URA": np.zeros(shape=(len(num_nodes), len(antenna_idx), num_simulations)),
        "iid": np.zeros(shape=(len(num_nodes), len(antenna_idx), num_simulations)),
    }

    pbar = tqdm(total=num_simulations * len(num_nodes))
    for sim in range(0, num_simulations):
        for n_idx, K in enumerate(num_nodes):

            for M in antenna_idx + 1:
                for conf, H_conf in zip(["ULA", "URA", "iid"], [H_ula, H_ura, None]):
                    if "iid" in conf:
                        channels = [normalize(np.sqrt(1 / 2) * (
                                np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M))) for n in
                                    range(K)]

                    else:
                        positions = list(range(0, len(H_conf)))
                        np.random.shuffle(positions)

                        rand_pos = positions[:K]

                        # only two frequencies
                        freq = random.randint(0, 1)

                        random.shuffle(antenna_idx)
                        random_antennas = antenna_idx[:M]

                        channels = []
                        for p in rand_pos:
                            h = H_conf[p]
                            time = random.randint(0, h.shape[0] - 1)
                            h = h[time, freq, random_antennas]
                            assert not np.isnan(h).any()
                            channels.append(h)

                    c = compute_condition_num(channels, M, K)
                    res[conf][n_idx][M - 1][sim] = c
            pbar.update()

    fig, ax = plt.subplots()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    colors = dict(zip(num_nodes, colors[:len(num_nodes)]))

    # axins = inset_axes(ax, width="40%", height=1.5, loc=9)
    for conf, k_values in res.items():
        for k, values in enumerate(k_values):
            avg_c = np.nanmean(values, axis=1)
            linestyle = "-"
            if conf == "ULA":
                linestyle = "--"
            elif conf == "iid":
                linestyle = "dotted"
            elif conf == "iid-p":
                linestyle = "-."
            ax.plot(np.arange(1, 32), avg_c, label=conf + " - " + str(num_nodes[k]), linewidth=1,
                    linestyle=linestyle, color=colors[num_nodes[k]])
            # axins.plot(avg_c, marker='o', label=conf, linewidth=2, markersize=3,
            #            alpha=0.5)

        # plt.plot(avg_c, label=conf, linestyle=linestyle)

    # x1, x2, y1, y2 = 4, 8, 0.3, 0.5  # specify the limits
    # axins.set_xlim(x1, x2)  # apply the x-limits
    # axins.set_ylim(y1, y2)  # apply the y-limits
    # from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    #
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.legend(ncol=4)

    from plotting import LatexifyMatplotlib as lm

    lm.save("condition-number-v3-small.tex", scale_legend=0.7, show=True, plt=plt)

    fig, ax = plt.subplots()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    colors = dict(zip(num_nodes, colors[:len(num_nodes)]))

    MAX_POINTS = 1000
    # axins = inset_axes(ax, width="40%", height=1.5, loc=9)
    for conf, k_values in res.items():
        for k, values in enumerate(k_values):
            ecdf = ECDF(values[30, :])

            num_points = len(ecdf.x)
            step = num_points//MAX_POINTS

            points = list(np.arange(0,num_points, step))
            if (num_points-1) not in points:
                points.append(num_points-1)


            linestyle = "-"
            if conf == "ULA":
                linestyle = "--"
            elif conf == "iid":
                linestyle = "dotted"
            elif conf == "iid-p":
                linestyle = "-."
            ax.plot(ecdf.x[points], ecdf.y[points], label=conf + " - " + str(num_nodes[k]), linewidth=1,
                    linestyle=linestyle,
                    alpha=0.5, color=colors[num_nodes[k]])

    plt.legend(ncol=3)
    _ = plt.xlabel('inverse cond number', size=14)
    _ = plt.ylabel('ECDF', size=14)

    from plotting import LatexifyMatplotlib as lm

    lm.save("condition-number-cdf-v3-small.tex", scale_legend=0.7, show=True, plt=plt)
