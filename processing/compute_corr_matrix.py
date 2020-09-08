import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_cov_matrix(d):
    path = pjoin(root_dir, d)
    input = pjoin(path, "small-channel.npy")

    if os.path.isfile(input):
        # [snapshots x freq points x BS antennas]
        H = np.load(input)

        # Correlation in MIMO Antennas - MDPI

        (N, F, M) = tuple(H.shape)

        R = np.zeros(shape=(M - 1, M - 1), dtype=complex)
        # Correlation in MIMO Antennas - MDPI
        # channel correlation at RX side is expressed as: H H^H
        # but as we average over freq and time it is the fading correlation
        for n in range(N):
            for f in range(F):
                h = H[n, f, :-1]
                # reshaping is transposing as 1D vector transpose is still a 1D vector in numpy
                h_herm = h.conjugate().reshape(-1, 1)
                # cast to 2D to be able to do the dot product
                h = np.atleast_2d(h)
                R = R + np.dot(h_herm, h)
        R = R / (N * F)

        # norm = np.linalg.norm(R, ord="fro")
        # # normalise R so frobenius norm is = M
        # # see paper "Multi-User Massive MIMO Properties in Urban-Macro Channel Measurements"
        # R = (R / norm) * M

        np.save(pjoin(path, "cov-matrix.npy"), R)

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
