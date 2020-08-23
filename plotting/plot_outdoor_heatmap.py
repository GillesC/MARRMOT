import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from tqdm import tqdm
import numpy as np
import pandas as pd
import folium
import branca
from folium import plugins
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import scipy as sp
import scipy.ndimage

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

    # pos_x = (pos_x - origin[0]) / delta_x
    # pos_y = (pos_y - origin[1]) / delta_x

    raw_evm = np.loadtxt(pjoin(meas, "raw-evm.txt"))[:, 0]
    if os.path.isfile(pjoin(meas, "small-channel.npy")):
        H = np.load(pjoin(meas, "small-channel.npy"))
    else:
        return None

    # [snapshots x freq points x BS antennas
    # remove faulty antenna 32
    H = np.sum(H[:,:,:-1], axis=2)
    H = 10 * np.log10(np.abs(H)^2)

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

        img = mpimg.imread(pjoin(current_path, "..", "img", "map-heatmap", "map.png"))

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

            contourf = plt.contourf(grid_x, grid_x, grid_h, alpha=0.5, linestyles='None',
                                    vmin=power_min, vmax=power_max)

            # Convert matplotlib contourf to geojson
            geojson = geojsoncontour.contourf_to_geojson(
                contourf=contourf,
                min_angle_deg=3.0,
                ndigits=5,
                stroke_width=1,
                fill_opacity=0.5)


            center = df.iloc[0]

            m = folium.Map(
                location=[center["latitude"], center["longitude"]],
                zoom_start=18,
            )

            colors = {
                "BS": "black",
                "A": "red",
                "B": "blue",
                "C": "green",
                "D": "purple"
            }

            for index, row in df.iterrows():
                popup = folium.Popup(
                    f'<a href="https://dramco.be/projects/marrmot/balcony/measurements/{row["path"]}-{int(row["point"])}-a-ULA-868/snapshots.html" target="_blank">{row["location"]} ULA</a><br>'
                    f'<a href="https://dramco.be/projects/marrmot/balcony/measurements/{row["path"]}-{int(row["point"])}-a-URA-868/snapshots.html" target="_blank">{row["location"]} URA</a>')
                # do not show popup with BS
                if index == 0:
                    popup = None

                folium.Marker(
                    icon=folium.Icon(color=colors[row["path"]]),
                    location=[row["latitude"], row["longitude"]],
                    popup=popup
                ).add_to(m)

            # Plot the contour plot on folium
            folium.GeoJson(
                geojson,
                style_function=lambda x: {
                    'color': x['properties']['stroke'],
                    'weight': x['properties']['stroke-width'],
                    'fillColor': x['properties']['fill'],
                    'opacity': 0.6,
                }).add_to(m)

            # Fullscreen mode
            plugins.Fullscreen(position='topright', force_separate_button=True).add_to(geomap)

            # Plot the data
            m.save(pjoin(current_path, f'heatmap_median_evm_{conf}.html'))


            # circle_radius = (np.max(x_arr) - np.min(x_arr)) / 100
            #
            # circles = [plt.Circle((p_x, p_y), circle_radius, fill=False) for p_x, p_y in
            #            zip(points_x[conf], points_y[conf])]
            #
            # plt.cla()
            # fig, ax = plt.subplots()
            # # plt.imshow(img, extent=[0, 1, 0, 1])
            # # [ax.add_artist(c) for c in circles]
            #
            # a = grid_h
            # a = np.ma.array(a, mask=np.isnan(a))
            # plt.imshow(a, extent=[np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)], origin='lower',
            #            vmin=power_min, vmax=power_max)
            # # [plt.scatter(0, 0, s=4000) for p_x, p_y in zip(points_x[conf], points_y[conf])]
            # [ax.add_artist(c) for c in circles]

            # plt.colorbar()
            # plt.savefig(pjoin(current_path, f'heatmap_median_h_{conf}.pdf'))
            #
            # plt.cla()
            # plt.subplots()
            # # plt.imshow(img, extent=[0, 1, 0, 1])
            # a = grid_evm
            # a = np.ma.array(a, mask=np.isnan(a))
            # plt.imshow(a, extent=[np.min(x_arr), np.max(x_arr), np.min(y_arr), np.max(y_arr)], origin='lower',
            #            cmap='viridis_r', vmin=evm_min, vmax=evm_max)
            # [ax.add_artist(c) for c in circles]
            # plt.colorbar()
            # plt.savefig(pjoin(current_path, f'heatmap_median_evm_{conf}.pdf'))
