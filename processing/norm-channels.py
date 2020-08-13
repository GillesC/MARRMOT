import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))


def norm_channel(d):
    path = os.path.join(root_dir, "measurements", d, "channel.npy")
    # [snapshots x freq points x BS antennas x users]
    H = np.load(path)
    N = H.shape[0]
    M = H.shape[2]

    H_norm = np.zeros(shape=(N, 2, M), dtype=complex)

    # used 52 instead of 51 because python works with an excluding stop index
    norm_factor = np.sqrt((1 / (N * M * 2)) * np.sum(np.abs(H[:, 50:52, :, 0]) ** 2))

    for n in range(N):
        # only use valuable frequency points
        for f, new_f in zip([50, 51], [0, 1]):
            # we only have one user
            H_norm[n, new_f, :] = H[n, f, :, 0] / norm_factor

    np.save(os.path.join(root_dir, "measurements", d, "norm-channel.npy"), H_norm)
    H_mat = {'H_norm': H_norm, 'H': H}
    sio.savemat(os.path.join(root_dir, "measurements", d, "norm-channel.mat"), H_mat)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=3) as executor:
        for d in dirs:
            future = executor.submit(norm_channel, d)
            future.add_done_callback(lambda p: pbar.update())
