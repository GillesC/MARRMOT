import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio

from utils.load_yaml import load_root_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def norm_channel(d):
    path = os.path.join(root_dir, d, "small-channel.npy")
    # [snapshots x freq points x BS antennas x users]
    H = np.load(path)
    N = H.shape[0]
    M = H.shape[2]

    H_norm = np.zeros(shape=(N, 2, M), dtype=complex)

    norm_factor = np.sqrt((1 / (N * M * 2)) * np.sum(np.abs(H) ** 2))

    for n in range(N):
        # only use valuable frequency points
        for f in [0, 1]:
            # we only have one user
            H_norm[n, f, :] = H[n, f, :] / norm_factor

    np.save(os.path.join(root_dir, d, "norm-channel.npy"), H_norm)
    H_mat = {'H_norm': H_norm, 'H': H}
    sio.savemat(os.path.join(root_dir, d, "norm-channel.mat"), H_mat)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=5) as executor:
        for d in dirs:
            future = executor.submit(norm_channel, d)
            future.add_done_callback(lambda p: pbar.update())
