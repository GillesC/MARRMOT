import scipy.io as sio
import os
from os.path import dirname, exists, abspath, join as pjoin
from load_yaml import *
import csv
from datetime import datetime
import numpy as np


root_dir = dirname(__file__)

conf = get_conf()

def divide_chunks(l, n): 
      
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "..", "data", env))
    if exists(env_path):
        print(f"{env}")
        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path,x)), os.listdir(env_path)):
            print(f"\t {scenario}")
            time_file = pjoin(env_path, scenario, "raw_timestamps.csv")
            if exists(time_file):
                with open(time_file, 'r') as stream:
                    print(f"\t\t Timestamps found")
                    timestamps = list(csv.reader(stream))
                    num_elem_per_chunk = 3
                    corrected_timestamps = []

                    corrected_timestamps_path = pjoin(env_path, scenario, "corrected_timestamps.npy")

                    for (snapshot_t, sys_t, ntp_t) in divide_chunks(timestamps, num_elem_per_chunk):
                        
                        snapshot_t = datetime.strptime(snapshot_t[0], '%I:%M:%S.%f %p %m/%d/%Y') #1:46:34.193 PM 3/12/2020
                        sys_t = datetime.strptime(sys_t[0], '%I:%M:%S.%f %p %m/%d/%Y') #1:46:34.193 PM 3/12/2020
                        ntp_t = datetime.strptime(ntp_t[0], '%I:%M:%S.%f %p %m/%d/%Y') #1:46:34.193 PM 3/12/2020

                        corrected_time = snapshot_t+ (sys_t-ntp_t)
                        corrected_time_str = corrected_time.strftime("%Y/%m/%d, %H:%M:%S.%f")
                        # print(corrected_time_str)
                        corrected_timestamps.append(corrected_time_str)

                    np.save(corrected_timestamps_path, corrected_timestamps)
                    print(f"\t\tCorrected {len(corrected_timestamps)} timestamps")
                           
                            



