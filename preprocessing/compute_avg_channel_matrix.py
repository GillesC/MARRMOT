import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio

from utils.load_yaml import load_root_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute(d):
    if is_cont_meas():
        return
    # [snapshots x freq points x BS antennas x users]
    H = np.load(os.path.join(root_dir, d, "small-channel.npy"))

    num_time = H.shape[0]
    num_freq = H.shape[1]

    avg_H = np.sum(H, axis=(0, 1)) / (num_time * num_freq)

    if not os.path.isfile(os.path.join(root_dir, d, "avg_H.npy")):
        np.save(os.path.join(root_dir, d, "avg_H.npy"), avg_H)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(compute, d)
            future.add_done_callback(lambda p: pbar.update())
