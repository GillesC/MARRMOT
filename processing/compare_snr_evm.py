import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import extract_info_from_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

evm_per_path = {
    'A': [],
    'B': [],
    'C': []
}


def get_evm(d):
    path = pjoin(root_dir, d)
    input_evm = pjoin(path, "raw-evm.txt")
    return np.loadtxt(input_evm, delimiter="\t")[:, 0]


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    for d in dirs:
        path, _, _, _, _ = extract_info_from_dir(d)
        evm_per_path[path].append(get_evm(d).tolist())
        pbar.update()


