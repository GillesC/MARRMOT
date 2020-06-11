import os
from os.path import exists

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.load_yaml import *

root_dir = abspath(pjoin(dirname(__file__), ".."))

conf = get_conf()

square_pos = {
    "UL": [2, 1],
    "UR": [2, 2],
    "LL": [1, 1],
    "LR": [1, 2]
}

for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")

        fig_full = make_subplots(rows=2, cols=2, start_cell="bottom-left")
        cmax = None
        cmin = None

        z_arr = []
        pos_arr = []

        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")
            scenario_dir_path = pjoin(env_path, scenario)
            input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
            df_channels = pd.read_feather(input_channels_path)
            df_channels = df_channels.query("(SubCarrier == 51 or SubCarrier == 52) and User == 0")



        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")

            scenario_dir_path = pjoin(env_path, scenario)
            input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
            df_channels = pd.read_feather(input_channels_path)

            df_channels = df_channels.query("SubCarrier == 51 and User == 0")
            x = df_channels["XPos"].unique()
            y = df_channels["YPos"].unique()

            x.sort()
            y.sort()

            X, Y = np.meshgrid(x, y)

            for ant in np.arange(1, conf["num_antennas"] + 1):
                Z = np.zeros(shape=(len(x), len(y)))
                for x_idx, x_pos in enumerate(x):
                    for y_idx, y_pos in enumerate(y):
                        power = df_channels.query(f"XPos == {x_pos} and YPos == {y_pos} and Antenna=={ant}")[
                            "ChannelPower_dB"].mean()
                        Z[x_idx, y_idx] = power
                        if cmax is None or power > cmax:
                            cmax = power
                        if cmin is None or power < cmin:
                            cmin = power
                z_arr.append(Z)
                pos_arr.append((square_pos[scenario][0], square_pos[scenario][1]))

        for z, (row, col) in zip(z_arr, pos_arr):
            data = go.Heatmap(z=z, zmin=cmin, zmax=cmax, showscale=True, visible=False, name=f"{ant}",
                              colorbar=dict(
                                  title="channel gain (dB)"
                              ),
                              colorscale="Viridis")
            fig_full.add_trace(data, row=row, col=col)

        fig_full.data[0].visible = True
        fig_full.data[conf["num_antennas"]].visible = True
        fig_full.data[2 * conf["num_antennas"]].visible = True
        fig_full.data[3 * conf["num_antennas"]].visible = True

        # Create and add slider
        steps = []
        for i in range(conf["num_antennas"]):
            step = dict(
                method="restyle",
                args=["visible", [False] * len(fig_full.data)],
                label=i + 1
            )
            step["args"][1][i] = True  # Toggle i'th trace to "visible"
            step["args"][1][i + conf["num_antennas"]] = True
            step["args"][1][i + 2 * conf["num_antennas"]] = True
            step["args"][1][i + 3 * conf["num_antennas"]] = True
            steps.append(step)
        steps.reverse()  # antenna 1 is at the right side
        sliders = [dict(
            currentvalue={"prefix": "Antenna: "},
            pad={"t": 50},
            steps=steps
        )]

        fig_full.update_layout(
            sliders=sliders
        )

        fig_full.update_yaxes(autorange="reversed")

        fig_full.update_layout(
            yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis2=dict(scaleanchor="x2", scaleratio=1),
            yaxis3=dict(scaleanchor="x3", scaleratio=1),
            yaxis4=dict(scaleanchor="x4", scaleratio=1),

        )

        fig_full.write_html(F"heatmap_all_antennas.html")
