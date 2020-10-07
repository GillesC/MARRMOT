import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
import warnings

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def sir(channels: np.ndarray, M, K, precoder):
    # K x M
    sir = np.zeros(K)

    assert precoder in ["ZF", "MR"]

    if precoder == "ZF":
        precoders = zf_precoder(channels.T, M, K)

    for k in range(K):
        h_k = channels[k]
        others = list(range(K))
        others.remove(k)  # remove user
        h_others = channels[others]
        if precoder == "MR":
            v_k = mr_precoder(h_k)
        elif precoder == "ZF":
            v_k = precoders[k]
        else:
            ValueError("Wrong argument for precoder")

        v_k_H = v_k.conj().T

        if np.all((v_k == 0)):
            sir_k = 0
        else:
            sir_k = np.abs(v_k_H @ h_k) ** 2 / np.sum([np.abs(v_k_H @ h_i) ** 2 for h_i in h_others])
        sir[k] = sir_k

    return sir


def mr_precoder(channel):
    # the MR precoder is just v_k = h_k/ ||h_k||
    if np.linalg.norm(channel) == 0:
        return channel
    return channel / np.linalg.norm(channel)


# def zf_precoder_old(H, M, K):
#     assert H.shape == (M, K)
#     # the MR precoder is just v_k = H(H^H H)^-1 / norm
#     pc = H @ np.linalg.inv((H.conj().T @ H))
#     # normalize per column, i.e. per user
#     for k in range(K):
#         pc[:, k] /= np.linalg.norm(pc[:, k])
#     return pc.T  # transpose so we return a KxM matrix


def zf_precoder(H, M, K):
    assert H.shape == (M, K)
    # if M == 1:
    #     # if num antennas is == 1 it is just MRC
    #     return np.array([mr_precoder(h) for h in H.T])

    # the MR precoder is just v_k = H(H^H H)^-1 / norm
    pc = H @ np.linalg.pinv(H.conj().T @ H)
    # normalize per column, i.e. per user
    for k in range(K):
        pc[:, k] /= np.linalg.norm(pc[:, k])
    return pc.T  # transpose so we return a KxM matrix


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
        H = H[:, :, :31]
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
    num_nodes = [2, 10]

    H_ula, H_ura = load_channels(dirs)
    precoders = ["ZF","MR"]
    # preallocate the results
    res = {
        "ULA": np.zeros(shape=(len(precoders), len(num_nodes), len(antenna_idx), num_simulations)).tolist(),
        "URA": np.zeros(shape=(len(precoders), len(num_nodes), len(antenna_idx), num_simulations)).tolist(),
        "iid": np.zeros(shape=(len(precoders), len(num_nodes), len(antenna_idx), num_simulations)).tolist(),
    }

    pbar = tqdm(total=num_simulations * len(num_nodes) * len(precoders))
    for prec_idx, prec in enumerate(precoders):
        for sim in range(0, num_simulations):
            for n_idx, K in enumerate(num_nodes):
                for M in antenna_idx + 1:
                    for conf, H_conf in zip(["ULA", "URA", "iid"], [H_ula, H_ura, None]):
                        if "iid" in conf:
                            channels = [np.sqrt(1 / 2) * (
                                    np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)) for n in
                                        range(K)]
                        else:
                            positions = list(range(0, len(H_conf)))
                            np.random.shuffle(positions)

                            # rand_pos = positions[:K]

                            # only two frequencies
                            freq = random.randint(0, 1)

                            random.shuffle(antenna_idx)
                            random_antennas = antenna_idx[:M]

                            channels = []
                            num_pos_found = 0

                            while num_pos_found < K:
                                assert len(positions) > 0
                                p = positions.pop()
                                h_full = np.array(H_conf[p])

                                pos_time = list(range(h_full.shape[0]))
                                random.shuffle(pos_time)

                                time = random.randint(0, h_full.shape[0] - 1)
                                h = h_full[time, freq, random_antennas]
                                channels.append(h)
                                num_pos_found += 1
                                # t_cnt = 0
                                # while np.linalg.norm(h) == 0 and len(pos_time) > 0:
                                #     # non-usable channel, look for another one
                                #     time = pos_time.pop()
                                #     h = h_full[time, freq, random_antennas]
                                # if len(pos_time) == 0:
                                #     # found no possible time instance where channel norm is not == 0
                                #     # try with another positions
                                #     pass
                                # else:
                                #     channels.append(h)
                                #     num_pos_found += 1

                        c = sir(np.array(channels), M, K, prec)
                        res[conf][prec_idx][n_idx][M - 1][sim] = c
                pbar.update()

    fig, ax = plt.subplots()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # axins = inset_axes(ax, width="40%", height=1.5, loc=9)
    for conf, prec_values in res.items():
        for precoder, k_values in enumerate(prec_values):
            for k, values in enumerate(k_values):
                avg_c = np.zeros(shape=len(antenna_idx))
                for m in range(len(antenna_idx)):
                    avg_c[m] = 10 * np.log10(np.nanmedian(np.array(values)[m, :, :]))
                avg_c = avg_c[np.isfinite(avg_c)]
                linestyle = "-"
                if conf == "ULA":
                    linestyle = "--"
                elif conf == "iid":
                    linestyle = "dotted"
                ax.plot(np.arange(31) + 1, avg_c, label=conf + " - " + precoders[precoder] + " - " + str(num_nodes[k]),
                        linewidth=1,
                        linestyle=linestyle, color=colors[precoder])
    plt.grid()
    plt.legend()
    plt.show()
