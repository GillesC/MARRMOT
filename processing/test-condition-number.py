import os
import random

import numpy as np

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula
import matplotlib.pyplot as plt

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_condition_num(H_list, M, K):
    H = np.zeros(shape=(M, K), dtype=complex)
    for k in range(K):
        H[:, k] = H_list[k]

    # assert H.shape == (M, K)

    H_H = np.conjugate(H).T

    # assert H_H.shape == (K, M)

    H_corr = H_H @ H

    # assert H_corr.shape == (K, K)

    u,s,vh = np.linalg.svd(H_corr)

    # eigvals = np.abs(np.linalg.eigvals(H_corr))
    # max_eigval = np.max(eigvals)
    # min_eigval = np.min(eigvals)
    #
    # if min_eigval == 0:
    #     return 0
    #
    # return min_eigval / max_eigval

    return np.min(s)/np.max(s)


if __name__ == '__main__':
    num_simulations = 10000
    num_nodes = [2, 4, 10]
    res = np.zeros(shape=(len(num_nodes), 31, num_simulations))

    for sim in range(0, num_simulations):
        for n_idx, K in enumerate(num_nodes):
            for M in range(1, 32):
                channels = [np.sqrt(1 / 2) * (
                        np.random.randn(M) + 1j * np.random.randn(M)) for n in
                            range(K)]

                c = compute_condition_num(channels, M, K)
                res[n_idx, M - 1, sim] = c

    fig, ax = plt.subplots()
    for n_idx, n in enumerate(num_nodes):
        avg_c = np.mean(res[n_idx], axis=1)
        ax.plot(np.arange(1, 32), avg_c, label=n, linewidth=1,
                alpha=0.5)
    plt.legend()
    plt.show()
