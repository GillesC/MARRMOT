import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute(d):
    input_evm = pjoin(pjoin(root_dir, d), "raw-evm.txt")
    raw_evm = np.loadtxt(input_evm, delimiter="\t")[:, 0]

    snr = 20*np.log10(100/raw_evm)

    np.save(os.path.join(root_dir, d, "snr.npy"), snr)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(compute, d)
            future.add_done_callback(lambda p: pbar.update())
