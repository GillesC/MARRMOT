import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import pandas as pd

from utils.load_yaml import load_root_dir
from utils.util_loc import extract_info_from_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def avg_h(d):
    # [BS antennas]
    H = np.load(os.path.join(root_dir, d, "avg_H.npy"))

    path, point, num, conf, freq = extract_info_from_dir(d)

    return [path, point, conf, freq, H]


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    data = []
    for d in dirs:
        data.append(avg_h(d))
        pbar.update()
    df = pd.DataFrame(data, columns=['path', 'point', 'conf', 'freq', 'avg_h'])
    print(df)
