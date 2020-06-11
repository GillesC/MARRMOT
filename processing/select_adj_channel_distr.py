import os
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import measure
import distributions as dist
import scipy.stats as st

from utils.load_yaml import *
from utils import pd_utils

root_dir = abspath(pjoin(dirname(__file__), ".."))

conf = get_conf()

square_pos = {
    "UL": [2, 1],
    "UR": [2, 2],
    "LL": [1, 1],
    "LR": [1, 2]
}

RECTANGLE = True


def largest_island(grid):
    labels = measure.label(grid, connectivity=1)
    regions = [region for region in measure.regionprops(labels)]
    areas = [region.area for region in regions]
    coordinates = regions[int(np.argmax(areas))].coords
    shape = grid.shape
    _res = np.zeros(shape=(shape[0], shape[1]))
    for coordinate in coordinates:
        _res[coordinate[0]][coordinate[1]] = 1
    return coordinates, _res


for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")

        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")

            scenario_dir_path = pjoin(env_path, scenario)
            input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
            df_channels = pd.read_feather(input_channels_path)

            df_channels = df_channels.query("(SubCarrier == 51 or SubCarrier == 52) and User == 0")
            max_idx = df_channels.ChannelPower_dB.argmin()
            xpos_max = df_channels.iloc[max_idx].XPos
            ypos_max = df_channels.iloc[max_idx].YPos
            mean_max = df_channels.query(f"XPos == {xpos_max} and YPos == {ypos_max}").ChannelPower_dB.mean()
            print(f"MAX: {df_channels.ChannelPower_dB.max()}")
            print(f"MIN: {df_channels.ChannelPower_dB.min()}   POS ({xpos_max},{ypos_max}) with mean: {mean_max}")

            # PER ANTENNA
            # for ant in range(1, conf["num_antennas"] + 1):
            #     df_selected_antenna = df_channels.query(f"Antenna=={ant}")
            #     # cast to int64 to be able to do bin count but first remove nans
            #     power = df_selected_antenna["ChannelPower_dB"].dropna()
            #     selected_antenna_powers = df_selected_antenna["ChannelPower_dB"].dropna().to_numpy(np.int32)
            #     selected_antenna_powers_shifted = selected_antenna_powers - np.nanmin(selected_antenna_powers)
            #
            #
            #     most_common_power = selected_antenna_powers[
            #         np.argmax(np.bincount(selected_antenna_powers_shifted))]
            #     # TODO change hard coded 3dB diff.
            #     potential_candidates = df_selected_antenna.query(
            #         f"ChannelPower_dB <= {most_common_power + 1.5} and ChannelPower_dB >= {most_common_power - 1.5}")
            #     # todo remove hardcoded 26
            #     x = np.arange(0, 1250 + 50, 50)
            #     y = np.arange(0, 1250 + 50, 50)
            #     z = np.zeros(shape=(26, 26))
            #
            #     print(
            #         f"Evaluating {len(potential_candidates.index)} candidates for antenna {ant} of total {len(df_selected_antenna.index)}")
            #
            #     for x_idx, x_pos in enumerate(x):
            #         for y_idx, y_pos in enumerate(y):
            #             # check per potential candidiate how many are there at each position
            #             res = potential_candidates.query(f"XPos == {x_pos} and YPos == {y_pos}")
            #             if not (res is None or res.empty):
            #                 z[x_idx, y_idx] = 1
            #
            #     coordinates, grid = largest_island(z)
            #
            #     best = None
            #
            #     # now plot the distr of island positions
            #     _ch_power_selected_pos = []
            #     for coordinate in coordinates:
            #         xpos = x[coordinate[0]]
            #         ypos = y[coordinate[1]]
            #         _ch_power_selected_pos = np.append(_ch_power_selected_pos, potential_candidates.query(f"XPos == {xpos} and YPos == {ypos}")["ChannelPower_dB"].tolist())
            #     data = pd.Series(_ch_power_selected_pos).dropna()

            best_dist = None

            # for ant in range(1, conf["num_antennas"] + 1):
            for ant in [0]:
                # df_selected_antenna = df_channels.query(f"Antenna=={ant}")
                df_selected_antenna = df_channels
                selected_antenna_powers = df_selected_antenna["ChannelPower_dB"].dropna().to_numpy(np.int32)

                selected_antenna_powers_shifted = selected_antenna_powers - np.nanmin(selected_antenna_powers)

                most_common_power = selected_antenna_powers[
                    np.argmax(np.bincount(selected_antenna_powers_shifted))]

                # combine Xpos and Ypos
                # df_selected_antenna["Pos"] = [x + '-' + y for x, y in
                #                               zip(df_selected_antenna.XPos.values, df_selected_antenna.YPos.values)]
                df_selected_antenna["Pos"] = df_selected_antenna.XPos.apply(str) + '-' + df_selected_antenna.YPos.apply(str)
                df_average_powers_per_pos = df_selected_antenna.groupby(["Pos"], as_index=False).mean()
               # use the average power for a location to do the selection criteria
                # TODO change hard coded 3dB diff.
                potential_candidates_pos = df_average_powers_per_pos.query(
                    f"ChannelPower_dB <= {most_common_power + 1} and ChannelPower_dB >= {most_common_power - 1}")[
                    "Pos"].values

                # extract candidates for the considered positions
                potential_candidates = df_selected_antenna[df_selected_antenna["Pos"].isin(potential_candidates_pos)]

                # todo remove hardcoded 26
                x = np.arange(0, 1250 + 50, 50)
                y = np.arange(0, 1250 + 50, 50)
                z = np.zeros(shape=(26, 26))

                for x_idx, x_pos in enumerate(x):
                    for y_idx, y_pos in enumerate(y):
                        # check per potential candidiate how many are there at each position
                        res = potential_candidates.query(f"XPos == {x_pos} and YPos == {y_pos}")
                        if not (res is None or res.empty):
                            z[x_idx, y_idx] = 1

                coordinates, grid = largest_island(z)

                # Find rectangle around the island
                if RECTANGLE:
                    max_x = x.min()
                    min_x = x.max()
                    min_y = y.max()
                    max_y = y.min()

                    for coordinate in coordinates:
                        xpos = x[coordinate[0]]
                        ypos = y[coordinate[1]]

                        min_x = xpos if xpos < min_x else min_x
                        max_x = xpos if xpos > max_x else max_x

                        min_y = ypos if ypos < min_y else min_y
                        max_y = ypos if ypos > max_y else max_y

                    x_rect = np.arange(min_x, max_x + 50, 50)
                    y_rect = np.arange(min_y, max_y + 50, 50)

                    print(f"Found an island with {len(coordinates)} positions for antenna {ant}")
                    print(f"Rectangle around the island contains {len(x_rect) * len(y_rect)} positions for antenna {ant}")

                    # now plot the distr of island positions
                    _ch_power_selected_pos = []
                    _df_pot_arr = []
                    for xpos in x_rect:
                        for ypos in y_rect:
                            df_pot = df_selected_antenna.query(f"XPos == {xpos} and YPos == {ypos}")
                            _df_pot_arr.append(df_pot)
                            _ch_power_selected_pos = np.append(_ch_power_selected_pos,
                                                               df_pot["ChannelAmplitude"].dropna().tolist())
                else:
                    _ch_power_selected_pos = []
                    _df_pot_arr = []
                    for coordinate in coordinates:
                        xpos = x[coordinate[0]]
                        ypos = y[coordinate[1]]
                        df_pot = df_selected_antenna.query(f"XPos == {xpos} and YPos == {ypos}")
                        _df_pot_arr.append(df_pot)
                        _ch_power_selected_pos = np.append(_ch_power_selected_pos,
                                                           df_pot["ChannelAmplitude"].dropna().tolist())

                heatmap_matrix, (_, _) = pd_utils.pd_to_heatmap_matrix(df_channels)
                sns.heatmap(heatmap_matrix)
                plt.show()
                heatmap_matrix, (_, _) = pd_utils.pd_to_heatmap_matrix(pd.concat(_df_pot_arr), x=x, y=y)
                sns.heatmap(heatmap_matrix)
                plt.show()

                data = pd.Series(_ch_power_selected_pos).dropna()
                print(data.min())
                print(data.max())

                best_fit_dist, best_fit_params, best_sse, (freq, bin_center) = dist.best_fit_distribution(data)

                if best_dist is None:
                    best_dist = (best_fit_dist, best_fit_params, best_sse, data)
                elif best_sse < best_dist[2]:
                    best_dist = (best_fit_dist, best_fit_params, best_sse, data)
                    print(f"New best distr found with sse {best_sse} ({best_fit_dist.name})")

            plt.figure(figsize=(12, 8))
            data = best_dist[-1]

            k = -1
            pdf = dist.make_pdf(best_dist[0], best_dist[1])
            if best_dist[0] is st.rice:
                k = dist.get_k_factor(best_dist[1])
                k = 10*np.log10(k)
                print(f"K = {k}dB")

            # Display
            plt.figure(figsize=(12, 8))
            ax = pdf.plot(lw=2, label=f'PDF ({best_dist[0].name}, {best_dist[1]},{best_dist[2]})', legend=True)
            data.plot(kind='hist', bins=len(bin_center), density=True, alpha=0.5, label='Data', legend=True, ax=ax)
            ax.set_title(f'Channel Amplitude for scenario {scenario}.\n All Fitted Distributions K={k}')
            ax.set_xlabel(u'Channel Amplitude')
            ax.set_ylabel('Frequency')
            # plt.legend()
            plt.show()
