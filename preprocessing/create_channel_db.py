import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio
import pandas as pd
from numpy import linalg as LA

from utils.load_yaml import load_root_dir
from utils.util_loc import extract_info_from_dir

import matplotlib.pyplot as plt

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

max_simulations = 100


def sir(h: np.ndarray, h_others: np.ndarray):
    h = h.reshape(-1, 1)
    h_herm = np.conjugate(h).T

    interference = [np.abs(np.dot(h_herm, h_other.reshape(-1, 1))) ** 2 for h_other in h_others]

    interference = np.sum(interference)
    return np.asscalar(np.abs(np.dot(h_herm, h)) ** 2 / interference)


def avg_h(d):
    # [BS antennas]
    path = os.path.join(root_dir, d)

    path_meas, point, num, conf, freq = extract_info_from_dir(d)

    is_cont = int(point) == 0

    if is_cont:
        H_avg = None
    else:
        H_avg = np.load(pjoin(path, "avg_h.npy"))[:-1]

    snr = np.load(pjoin(path, "snr.npy"))
    H = np.load(pjoin(path, "small-channel.npy"))[:-1]

    return [path_meas, point, num, conf, freq, is_cont, H, H_avg, snr]


if __name__ == '__main__':

    data = []
    for d in dirs:
        res = avg_h(d)
        if res is not None:
            data.append(res)

    df = pd.DataFrame(data, columns=['path', 'point', 'num', 'conf', 'freq', 'is_cont', 'H', 'H_avg', "snr"]).astype({
        "path": "str",
        "point": "int",
        "num": "str",
        "conf": "str",
        "freq": "int",
        "is_cont": "bool",
    })
    df.to_hdf(pjoin(root_dir, "db.h5"), key="df", mode="w")

    del df

    df = pd.read_hdf(pjoin(root_dir, "db.h5"), key="df")
