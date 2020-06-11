import os
from os.path import exists
from time import perf_counter

import pandas as pd

from load_yaml import *

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

            print(f"\t\tReading channels", end='', flush=True)
            tic = perf_counter()
            _df = pd.read_feather(channel_file_path)
            toc = perf_counter()
            print(f"done {toc - tic}s")

            all_data_carriers = []

            for data_carrier in conf["data_carriers"]:
                print(f"\t\tExtracting data carrier {data_carrier}...", end='', flush=True)
                tic = perf_counter()
                res = _df.query(f"SubCarrier == {data_carrier}").reset_index()
                all_data_carriers.append(res)
                print(f"\t\t\t{res.shape}")
                _path = pjoin(scenario_dir_path, f"channels_data_carr_{data_carrier}.ftr")
                res.to_feather(_path)
                toc = perf_counter()
                print(f"done {toc - tic}s")

            print(f"\t\tStoring all data carriers...", end='', flush=True)
            tic = perf_counter()
            df_all_data_carriers = pd.concat(all_data_carriers, ignore_index=True, copy=False).reset_index()
            _path = pjoin(scenario_dir_path, f"channels_data_carr_all.ftr")
            df_all_data_carriers.to_feather(_path)
            toc = perf_counter()
            print(f"done {toc - tic}s")
