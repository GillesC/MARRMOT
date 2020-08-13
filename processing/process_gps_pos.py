from geopy.distance import distance
import pandas as pd
import os
import numpy as np
from os.path import join as pjoin

root_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = root_dir.replace("\\", "/")
df = pd.read_excel('GEO-data.xlsx')

df = df.dropna(how="all")

df.loc[:, "latitude"] = df["latitude"] / 10e6
df.loc[:, "longitude"] = df["longitude"] / 10e6

df.loc[:, "point"] = df["point"].astype(int)

ref_point = df.query("path == 'BS'")
ref_point = ref_point.iloc[0]

for index, row in df.iterrows():
    if np.isnan(row["latitude"]):
        continue

    path = row["path"]
    point = row["point"]
    dist = distance((ref_point["latitude"], ref_point["longitude"]), (row["latitude"], row["longitude"])).m
    df.loc[index, "distance"] = dist

df.to_csv(pjoin(root_dir, "gps-loc.csv"), index=False)
