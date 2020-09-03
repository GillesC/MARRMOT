import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm
import utils

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))


def compute_cmd(d):
    path, point, num, conf, freq = utils.extract_info_from_dir(d)

    # do not process first point or continuous point
    if point == 0 or point == 1:
        return

    R1 = np.load(pjoin(root_dir, "measurements", f"A-1-a-{conf}-{freq}", "cov-matrix.npy"))
    R2 = np.load(pjoin(root_dir, "measurements", d, "cov-matrix.npy"))

    R1 = np.asmatrix(R1)
    R2 = np.asmatrix(R2)

    d_cmd = np.trace(np.dot(R1.H, R2)) / (np.linalg.norm(R1, ord="fro") * np.linalg.norm(R2, ord="fro"))
    np.save(pjoin(root_dir, "measurements", d, "d_cmd.npy"), np.abs(d_cmd))


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(compute_cmd, d)
            future.add_done_callback(lambda p: pbar.update())
