import pandas as pd
from os.path import dirname, join as pjoin
import scipy.io as sio
import plotly.graph_objects as go
import numpy as np


root_dir = dirname(__file__)
data_dir = pjoin(root_dir, "..", "data")

positions_file = pjoin(data_dir, "test-exp1.csv")
channels_file = pjoin(data_dir, "channels.mat")

pd_positions = pd.read_csv(positions_file, sep="\t", header=0, parse_dates=[
                           "start", "stop"], infer_datetime_format=True)
# py_channels = sio.loadmat(channels_file)

pd_merged = pd.DataFrame(
    columns=["time", "x_pos", "y_pos", "channels", "avg_channel_power"])


merged_channels_arr = []
print(pd_positions.head())
for idx_pos, row_pos in pd_positions.iterrows():
    channels_for_current_pos = pd_channels.between_time(
        start_time=row_pos["start"], stop_time=row_pos["stop"])
    current_pos_x = row_pos["x"]
    current_pos_y = row_pos["y"]

    for idx_ch, row_ch in channels_for_current_pos.iterrows():
        merged_channels_arr.append({
            "time": row_ch["time"],
            "x_pos": current_pos_x,
            "y_pos": current_pos_y,
            "channels": row_ch["channels"],
            "avg_channel_power": np.mean(np.abs(row_ch["channels"])**2)
        })


fig = go.Figure(data=go.Heatmap(x=pd_merged["x_pos"], y=pd_merged["y_pos"], z=pd_merged["avg_channel_power"]))
fig.write_html("test-exp1.html")