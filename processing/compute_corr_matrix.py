import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

import seaborn as sns
import matplotlib.pylab as plt

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))


def compute_cov_matrix(d):
    path = pjoin(root_dir, "measurements", d)
    input = pjoin(path, "norm-channel.npy")
    # [snapshots x freq points x BS antennas]
    H_norm = np.load(input)

    (N, F, M) = tuple(H_norm.shape)

    R = np.zeros(shape=(M-1, M-1), dtype=complex)
    for n in range(N):
        for f in range(F):
            h = H_norm[n, f, :-1]
            # reshaping is transposing as 1D vector transpose is still a 1D vector in numpy
            h_herm = h.conjugate().reshape(-1, 1)
            # cast to 2D to be able to do the dot product
            h = np.atleast_2d(h)
            R = R + np.dot(h_herm, h)
    R = R / (N * F)
    norm = np.linalg.norm(R, ord="fro")
    # normalise R so frobenius norm is = M
    # see paper "Multi-User Massive MIMO Properties in Urban-Macro Channel Measurements"
    R = (R / norm) * M

    np.save(pjoin(path,"cov-matrix.npy"), R)

    # R_dB = 10 * np.log10(np.abs(R))
    #
    #
    #
    # plt.close('all')
    #
    # ax = sns.heatmap(R_dB, square=True, cmap='viridis')
    # fig = ax.get_figure()
    # fig.savefig(pjoin(path, "norm-covariance-without-last-antenna.png"))


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(compute_cov_matrix, d)
            future.add_done_callback(lambda p: pbar.update())
