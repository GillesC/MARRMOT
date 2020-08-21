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

points = []

data_path = "D:\Stack\measurement-data"
current_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(pjoin(current_path, "..", "gps-loc.csv"))

dirs = [f.path for f in os.scandir(data_path) if f.is_dir()]


def compute(meas):
    path, point, num, conf, freq = util_loc.extract_info_from_dir(meas)

    if point == 0 or conf != "ULA":
        return None

    loc = util_loc.get_meas(path, point)

    pos_x = loc["longitude"]
    pos_y = loc["latitude"]

    pos_x = (pos_x - origin[0]) / delta_x
    pos_y = (pos_y - origin[1]) / delta_x

    raw_evm = np.loadtxt(pjoin(meas, "raw-evm.txt"))[:, 0]
    if os.path.isfile(pjoin(meas, "norm-channel.npy")):
        H = np.load(pjoin(meas, "norm-channel.npy"))
    else:
        return None

    H = 10 * np.log10(np.abs(H))

    return np.median(raw_evm), np.median(H), (pos_x, pos_y)


evm_values = []
power_values = []
if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(compute, d) for d in dirs]

        for future in as_completed(futures):
            pbar.update()
            res = future.result()
            if res is not None:
                evm, H_median, (pos_x, pos_y) = res
                points.append([pos_y, pos_x])
                evm_values.append(evm)
                power_values.append(H_median)

        print("Done processing points")

        grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]

        grid_z0 = griddata(points, power_values, (grid_x, grid_y), method='nearest')
        grid_z1 = griddata(points, power_values, (grid_x, grid_y), method='linear')
        grid_z2 = griddata(points, power_values, (grid_x, grid_y), method='cubic')

        plt.subplot(221)
        img = plt.imread(pjoin(current_path, "..", "img", "map-heatmap", "map.png"))
        fig, ax = plt.subplots()
        ax.imshow(img, extent=[0, 1, 0, 1])
        plt.subplot(222)
        plt.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
        plt.title('Nearest')
        plt.subplot(223)
        plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
        plt.title('Linear')
        plt.subplot(224)
        plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
        plt.title('Cubic')
        plt.gcf().set_size_inches(6, 6)
        plt.show()

        plt.cla()
        fig, ax = plt.subplots()
        #plt.imshow(img, extent=[0, 1, 0, 1])
        a = grid_z2.T
        a = np.ma.array(a, mask=np.isnan(a))
        plt.imshow(a, extent=[0, 1, 0, 1], origin='lower')
        plt.colorbar()
        plt.savefig(pjoin(current_path,'heatmap_median_h.pdf'))

        grid_z0 = griddata(points, evm_values, (grid_x, grid_y), method='nearest')
        grid_z1 = griddata(points, evm_values, (grid_x, grid_y), method='linear')
        grid_z2 = griddata(points, evm_values, (grid_x, grid_y), method='cubic')

        plt.cla()
        fig, ax = plt.subplots()
        #plt.imshow(img, origin = 'lower',  extent = [0, img.shape[0], 0, img.shape[1]], aspect = 1000)
        a = grid_z1.T
        a = np.ma.array(a, mask=np.isnan(a))
        plt.imshow(a, extent=[0, 1, 0, 1], origin='lower')
        plt.colorbar()
        plt.savefig(pjoin(current_path,'heatmap_median_evm.pdf'))

