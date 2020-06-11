import scipy.io as sio
import os
from os.path import dirname, exists, abspath, join as pjoin
from utils.load_yaml import *
import csv
from datetime import datetime
import numpy as np
import pandas as pd
from dask import dataframe as dd
from time import perf_counter
import cmath
import matplotlib.pyplot as plt

preprocessing_dir = dirname(__file__)
root_dir = abspath(pjoin(preprocessing_dir, ".."))

conf = get_conf()


for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")
        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")

            scenario_dir_path = pjoin(env_path, scenario)
            channel_file_path = pjoin(scenario_dir_path, "channels.ftr")

            for data_carrier in conf["data_carriers"]:
                print(f"\t\tReading channels for SC {data_carrier}...", end='', flush=True)
                tic = perf_counter()
                _path = pjoin(scenario_dir_path, f"channels_data_carr_{data_carrier}.ftr")
                _df = pd.read_feather(_path)
                _df["Antenna"].astype(int)
                _df["User"].astype(int)
                _df = _df.query("Antenna == 15 and User == 0").reset_index()
                plt.title(f"ENV {scenario} SC {data_carrier}")
                plt.plot(_df["ChannelAmplitude"]**2)
                plt.show()
                toc = perf_counter()
                print(f"done {toc - tic}s")
