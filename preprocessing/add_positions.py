import os
from os.path import exists

import pandas as pd

from utils.load_yaml import *

root_dir = abspath(pjoin(dirname(__file__), ".."))

conf = get_conf()

for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")
        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")

            scenario_dir_path = pjoin(env_path, scenario)
            input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all.ftr")
            input_position_path = pjoin(scenario_dir_path, "positions.csv")

            output_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
            df_channels = pd.read_feather(input_channels_path)
            df_channels.loc[:, "Timestamp"] = pd.to_datetime(df_channels["Timestamp"], format="%Y/%m/%d, %H:%M:%S.%f")
            df_channels.index = pd.DatetimeIndex(df_channels["Timestamp"])


            pd_positions = pd.read_csv(input_position_path, sep="\t", header=0, infer_datetime_format=False)
            pd_positions.loc[:, "start"] = pd.to_datetime(pd_positions["start"], format="%Y-%m-%d %H:%M:%S.%f")
            pd_positions.loc[:, "stop"] = pd.to_datetime(pd_positions["stop"], format="%Y-%m-%d %H:%M:%S.%f")

            channels_with_positions = []
            for idx_pos, row_pos in pd_positions.iterrows():
                timestamp = df_channels["Timestamp"]
                channels_at_current_pos = df_channels[(timestamp >= row_pos["start"]) & (timestamp <= row_pos["stop"])]
                channels_at_current_pos.loc[:, "XPos"] = row_pos["x"]
                channels_at_current_pos.loc[:, "YPos"] = row_pos["y"]
                channels_with_positions.append(channels_at_current_pos)

            pd_channels_with_positions = pd.concat(channels_with_positions, ignore_index=True, copy=False).reset_index(
                drop=True)
            pd_channels_with_positions.to_feather(output_path)
