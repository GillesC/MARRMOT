import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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

points_x = {"ULA": [], "URA": []}
points_y = {"ULA": [], "URA": []}

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

    #pos_x = (pos_x - origin[0]) / delta_x
    #pos_y = (pos_y - origin[1]) / delta_x

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
                points_x[conf].append(pos_x)
                points_y[conf].append(pos_y)
                evm_values[conf].append(evm)
                power_values[conf].append(H_median)

        print("Done processing points")

        img = mpimg.imread(pjoin(current_path, "..","img","map-heatmap","map.png"))



        power_min = 100
        power_max = -100

        evm_min = 100
        evm_max = -1

        for conf in points_x.keys():
            if power_min > np.min(power_values[conf]):
                power_min = np.min(power_values[conf])
            if power_max < np.max(power_values[conf]):
                power_max = np.max(power_values[conf])

            if evm_min > np.min(evm_values[conf]):
                evm_min = np.min(evm_values[conf])
            if evm_max < np.max(evm_values[conf]):
                evm_max = np.max(evm_values[conf])

        for conf in points_x.keys():
            x_arr = np.linspace(np.min(points_x[conf]), np.max(points_x[conf]), 500)
            y_arr = np.linspace(np.min(points_y[conf]), np.max(points_y[conf]), 500)
            grid_x, grid_y = np.meshgrid(x_arr, y_arr)


            grid_h = griddata((points_x[conf], points_y[conf]), power_values[conf], (grid_x, grid_y), method='linear')
            grid_evm = griddata((points_x[conf], points_y[conf]), evm_values[conf], (grid_x, grid_y), method='linear')

            circles = [plt.Circle((p_x, p_y), 0.01, fill=False) for p_x, p_y in zip(points_x[conf], points_y[conf])]

            plt.cla()
            fig, ax = plt.subplots()
            #plt.imshow(img, extent=[0, 1, 0, 1])
            [ax.add_artist(c) for c in circles]
            a = grid_h.T
            a = np.ma.array(a, mask=np.isnan(a))
            plt.imshow(a, origin='lower', vmin=power_min, vmax=power_max)
            plt.colorbar()
            plt.savefig(pjoin(current_path, f'heatmap_median_h_{conf}.pdf'))

            plt.cla()
            plt.subplots()
            #plt.imshow(img, extent=[0, 1, 0, 1])
            a = grid_evm.T
            a = np.ma.array(a, mask=np.isnan(a))
            plt.imshow(a, origin='lower', cmap='viridis_r', vmin=evm_min, vmax=evm_max)
            plt.colorbar()
            plt.savefig(pjoin(current_path, f'heatmap_median_evm_{conf}.pdf'))
