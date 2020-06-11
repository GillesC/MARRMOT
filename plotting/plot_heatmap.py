import pandas as pd
from os.path import dirname, join as pjoin
import scipy.io as sio
import plotly.graph_objects as go
import numpy as np
import os
from os.path import exists
from plotly.subplots import make_subplots

import pandas as pd

from utils.load_yaml import *


root_dir = abspath(pjoin(dirname(__file__), ".."))

conf = get_conf()

square_pos = {
    "UL": [2, 1],
    "UR": [2, 2],
    "LL": [1, 1],
    "LR": [1, 2]
}
go.Layout(
    yaxis=dict(autorange='reversed')
)

for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")

        fig_full = make_subplots(rows=2, cols=2, start_cell="bottom-left")

        cmin = 1
        cmax = -1

        traces = {}
        showscale = True

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

            Z = np.zeros(shape=(len(x), len(y)))

            for x_idx, x_pos in enumerate(x):
                for y_idx, y_pos in enumerate(y):
                    power = df_channels.query(f"XPos == {x_pos} and YPos == {y_pos} and Antenna==17")[
                        "ChannelPower_dB"].mean()
                    Z[x_idx, y_idx] = power

            traces[scenario] = {
                "z": Z,
                "row": square_pos[scenario][0],
                "col": square_pos[scenario][1]
            }
            _current_max = Z.max()
            _current_min = Z.min()

            cmax = cmax if _current_max < cmax else _current_max
            cmin = cmin if _current_min > cmin else _current_min

        for scenario, trace in traces.items():
            _heatmap = go.Heatmap(z=trace["z"], zmin=cmin, zmax=cmax, showscale=showscale)
            showscale = False
            fig_full.add_trace(_heatmap, row=trace["row"], col=trace["col"])
            fig = go.Figure(data=[_heatmap])
            fig.update_yaxes(autorange="reversed")
            fig.write_html(F"heatmap_{scenario}.html")

        fig_full.update_yaxes(autorange="reversed")

        fig_full.update_layout(

            yaxis=dict(scaleanchor="x", scaleratio=1),
            yaxis2=dict(scaleanchor="x2", scaleratio=1),
            yaxis3=dict(scaleanchor="x3", scaleratio=1),
            yaxis4=dict(scaleanchor="x4", scaleratio=1),

        )

        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=[{"type": ["surface", "surface"]},
                                  {'traces': [0, 1, 2, 3]}],  # restyle both fig.data[0], fig.data[1] to "surface"
                            label="3D Surface",
                            method="restyle"
                        ),
                        dict(
                            args=[{"type": ["heatmap", "heatmap"]},
                                  {'traces': [0, 1, 2, 3]}],  # restyle both fig.data[0], fig.data[1] to "surface"
                            label="Heatmap",
                            method="restyle"
                        ),

                    ],
                    # direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1,
                    xanchor="left",
                    y=1,
                    yanchor="top"
                )
            ]
        )

        fig_full.write_html(F"heatmap_all.html")
