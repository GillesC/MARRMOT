import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm

from utils import util_loc

# normalize positions

origin = (4.68290, 50.86097)
corner = (4.68669, 50.86327)

delta_x = abs(origin[0] - corner[0])
delta_y = abs(origin[1] - corner[1])

points = {"ULA": [], "URA": []}

evm_values = {"ULA": [], "URA": []}
power_values = {"ULA": [], "URA": []}

data_path = "D:\Stack\measurement-data"
current_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(pjoin(current_path, "..", "gps-loc.csv"))

dirs = [f.path for f in os.scandir(data_path) if f.is_dir()]


def compute(meas):
    path, point, num, conf, freq = util_loc.extract_info_from_dir(meas)

    if point == 0:
        return None

    loc = util_loc.get_meas(path, point)

    pos_x = loc["longitude"]
    pos_y = loc["latitude"]

    pos_x = (pos_x - origin[0]) / delta_x
    pos_y = (pos_y - origin[1]) / delta_x

    raw_evm = np.loadtxt(pjoin(meas, "raw-evm.txt"))[:, 0]
    if os.path.isfile(pjoin(meas, "small-channel.npy")):
        H = np.load(pjoin(meas, "small-channel.npy"))
    else:
        return None

    H = 10 * np.log10(np.abs(H))

    return np.median(raw_evm), np.median(H), (pos_x, pos_y), conf, freq


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(compute, d) for d in dirs]

        for future in as_completed(futures):
            pbar.update()
            res = future.result()
            if res is not None:
                evm, H_median, (pos_x, pos_y), conf, _ = res
                points[conf].append([pos_y, pos_x])
                evm_values[conf].append(evm)
                power_values[conf].append(H_median)

        print("Done processing points")

        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

        for conf in points.keys():
            grid_h = griddata(points[conf], power_values[conf], (grid_x, grid_y), method='linear')
            grid_evm = griddata(points[conf], evm_values[conf], (grid_x, grid_y), method='linear')

            plt.cla()
            plt.subplots()
            # plt.imshow(img, extent=[0, 1, 0, 1])
            a = grid_h.T
            a = np.ma.array(a, mask=np.isnan(a))
            plt.imshow(a, extent=[0, 1, 0, 1], origin='lower')
            plt.colorbar()
            plt.savefig(pjoin(current_path, f'heatmap_median_h_{conf}.pdf'))


            plt.cla()
            plt.subplots()
            # plt.imshow(img, extent=[0, 1, 0, 1])
            a = grid_evm.T
            a = np.ma.array(a, mask=np.isnan(a))
            plt.imshow(a, extent=[0, 1, 0, 1], origin='lower', cmap='viridis_r')
            plt.colorbar()
            plt.savefig(pjoin(current_path, f'heatmap_median_evm_conf.pdf'))
