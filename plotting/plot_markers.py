import glob

import folium
import pandas as pd

from utils import util_loc
from utils.load_yaml import load_root_dir
from os.path import join as pjoin
import os

root_dir = load_root_dir()
df = pd.read_csv('../gps-loc.csv')

center = df.iloc[0]

m = folium.Map(
    location=[center["latitude"], center["longitude"]],
    zoom_start=18,
)

colors = {
    "BS": "black",
    "A": "red",
    "B": "blue",
    "C": "green",
    "D": "purple"
}

for index, row in df.iterrows():
    meas_path = row["path"]
    meas_point = int(row["point"])
    meas_location = row["location"]

    popup_text = f"{meas_location}<br>"
    for d in glob.glob(pjoin(root_dir, f"{meas_path}-{meas_point}-a-*-*")):
        _, _, _, conf, freq = util_loc.extract_info_from_dir(d)
        popup_text += f'<a href="https://dramco.be/projects/marrmot/balcony/measurements/{os.path.basename(d)}/snapshots.html" target="_blank">{conf} {freq}</a><br> '

    popup = folium.Popup(popup_text, max_width=200)

    # do not show popup with BS
    if index == 0:
        popup = None

    folium.Marker(
        icon=folium.Icon(color=colors[row["path"]]),
        location=[row["latitude"], row["longitude"]],
        popup=popup
    ).add_to(m)

m.save(pjoin(root_dir, "index.html"))
