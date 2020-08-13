import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import os
from os.path import join as pjoin

# normalize positions

origin = (4.68290, 50.86097)
corner = (4.68669, 50.86327)

points = []
evm_values = []
power_values = []

data_path = "I:\MARRMOT\measurements"

subfolders = [f.path for f in os.scandir(data_path) if f.is_dir()]

for meas in subfolders:
    delta_x = abs(origin[0] - corner[0])
    delta_y = abs(origin[1] - corner[1])

    pos_x = (pos_x - origin[0])/delta_x
    pos_y = (pos_y - origin[1])/delta_x

    points.append([pos_x, pos_y])
    raw_evm = np.loadtxt(pjoin(meas, "raw-evm.txt"))[:, 0]
    H = np.load(pjoin(meas, "channel.npy"))
    evm_values.append(np.median(raw_evm))
    power_values.append(np.median(H))


grid_x, grid_y = np.mgrid[0:1:20j, 0:1:20j]


grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

plt.subplot(221)
img = plt.imread(pjoin(data_path, "..", "img","map-heatmap","map.png"))
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
