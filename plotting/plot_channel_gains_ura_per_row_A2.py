import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = load_root_dir()
    subdir, dirs, files = next(os.walk(os.path.join(root_dir)))
    H = np.load(os.path.join(root_dir, "A-2-a-URA-868", "small-channel.npy"))
    gain = np.abs(H) ** 2
    mean = np.mean(10 * np.log10(gain), axis=0)

