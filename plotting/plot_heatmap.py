import pandas as pd
from os.path import dirname, join as pjoin
import scipy.io as sio
import plotly.graph_objects as go
import numpy as np

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, "..", "data")
channels_file = pjoin(data_dir, "H.mat") #MxfxUExPos

py_channels = sio.loadmat(channels_file)
channel = py_channels["H_UE"]




for idx_bs, bs in enumerate(channel):
    num_users = len(bs)
    print(F"Plotting {num_users} users")

    power = []
    for idx_user, user_in_bs in enumerate(bs):
        print(F"{idx_user}/{len(bs)}")
        print(user_in_bs["PperF"][0])
        power.append(user_in_bs["PperF"][0])

    power = np.array(power).flatten()

    power = np.reshape(power, (int(np.sqrt(num_users)), int(np.sqrt(num_users))))
    print(power)
    data = go.Heatmap(z=power)
    fig = go.Figure(data=data)
    fig.write_html(F"heatmap-BS{idx_bs}.html")
