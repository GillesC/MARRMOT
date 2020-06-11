import cmath
import csv
import os
from os.path import exists
from time import perf_counter

import numpy as np
import pandas as pd

from load_yaml import *

preprocessing_dir = dirname(__file__)
root_dir = abspath(pjoin(preprocessing_dir, ".."))

conf = get_conf()


def divide_chunks(l, n):
    # looping till length l 
    for i in range(0, len(l), n):
        yield l[i:i + n]


for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")
        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")

            scenario_dir_path = pjoin(env_path, scenario)
            imag_file_path = pjoin(scenario_dir_path, "raw_imag_samples.csv")
            real_file_path = pjoin(scenario_dir_path, "raw_real_samples.csv")

            if exists(imag_file_path):

                corrected_timestamps_path = pjoin(env_path, scenario, "corrected_timestamps.npy")
                assert exists(corrected_timestamps_path), "Need timestamps yoo, where you stored them timestamps"
                print(f"\t\tLoading Timestamps...", end='')
                tic = perf_counter()
                corrected_timestamps = np.load(corrected_timestamps_path)
                toc = perf_counter()
                print(f"loaded {toc - tic}s")

                num_channel_captures = len(corrected_timestamps)
                print(f"\t\t{num_channel_captures} captures found")

                # create antenna number for each entry
                # every "num_sub_carriers"th subcarrier a new antenna entry begins
                # after num_sub_carriers*num_antennas -> a new channel capture begins
                # [ [][][]...[] ] [ [][][]...[] ] [ [][][]...[] ]
                #   ^
                #   |_ 100 subcarriers for antenna 1       
                # |             | <- first channel capture 

                print(f"\t\tGenerating indices...", end='')
                tic = perf_counter()
                antenna_idx = np.arange(1, conf["num_antennas"] + 1)  # +1 because exclusive
                # repeat antenna idx for every subcarrier (per element)
                antenna_idx = np.repeat(antenna_idx, conf["num_sub_carriers"])
                # repeat antenna idx for every channel capture (whole array)
                antenna_idx = np.tile(antenna_idx, num_channel_captures)

                subcarrier_idx = np.arange(1, conf["num_sub_carriers"] + 1)
                subcarrier_idx = np.tile(subcarrier_idx, num_channel_captures * conf["num_antennas"])

                # we have the timestamp er channel capture
                # so this is for all antennas and subcarriers at once
                # so the same timestamp for ever NUM_SC * NUM_ANT
                corrected_timestamps = np.repeat(corrected_timestamps, conf["num_sub_carriers"] * conf["num_antennas"])

                toc = perf_counter()
                print(f"done {toc - tic}s")

                num_users = 2  # TODO read out file

                # channel_captures = np.empty(num_channel_captures*num_users*conf["num_antennas"]*conf["num_sub_carriers"], dtype=object) # pre-allocating
                channel_captures = []
                channel_vals = []

                print(f"\t\tProcessing files for {env}/{scenario}...", end='', flush=True)
                tic = perf_counter()
                ant_mapping = conf["ant_to_cable_mapping"]
                with open(imag_file_path, newline='') as imag_file, open(real_file_path, newline='') as real_file:
                    imag_reader = csv.reader(imag_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
                    real_reader = csv.reader(real_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

                    labels = ["Antenna", "SubCarrier", "User", "Timestamp", "ChannelAmplitude", "ChannelPhase",
                              "ChannelPower_dB"]
                    for line_idx, (imag, real, a_idx, sc_idx, time) in enumerate(zip(imag_reader, real_reader, antenna_idx, subcarrier_idx,
                                                               corrected_timestamps)):
                        for u_idx, (imag_u, real_u) in enumerate(zip(imag, real)):

                            (r, phi) = cmath.polar(complex(real_u, imag_u))
                            # channel_captures[i] = (a_idx, sc_idx,u_idx,time,val)
                            assert r < 1, f"Channel gain should be < 1 but is {r} at line {line_idx} with channel {real_u}+j{imag_u}"
                            r_db = np.nan if r == 0 else 20 * np.log10(r)

                            channel_captures.append((ant_mapping[a_idx], sc_idx, u_idx, time, r, phi, r_db))

                            # channel_captures.append({
                            #     "Antenna":a_idx,
                            #     "SubCarrier": sc_idx,
                            #     "User": u_idx,
                            #     "Timestamp": time,
                            #     "Channel":real_u+1j*imag_u
                            # })

                    toc = perf_counter()
                    print(f"done {toc - tic}s")

                # clean-up
                del antenna_idx
                del subcarrier_idx
                del corrected_timestamps

                print(f"\t\tStoring channel...", end='', flush=True)
                tic = perf_counter()
                output_file_path = pjoin(scenario_dir_path, "channels.ftr")
                _df = pd.DataFrame.from_records(channel_captures, columns=labels)
                # "Antenna", "SubCarrier", "User","Timestamp","ChannelAmplitude","ChannelPhase"

                _df.astype({"Antenna": "int", "SubCarrier": "int", "User": "int", "Timestamp": "str",
                            "ChannelAmplitude": "float", "ChannelPhase": "float", "ChannelPower_dB": "float"})

                del channel_captures

                _df.to_feather(output_file_path)
                toc = perf_counter()
                print(f"done {toc - tic}s")
