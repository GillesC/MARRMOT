import folium
import pandas as pd
import os

root_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = root_dir.replace("\\", "/")
df = pd.read_csv('gps-loc.csv')

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
    popup = folium.Popup(
        f'<a href="https://dramco.be/projects/marrmot/balcony/measurements/{row["path"]}-{int(row["point"])}-a-ULA-868/snapshots.html" target="_blank">{row["location"]} ULA</a><br>'
        f'<a href="https://dramco.be/projects/marrmot/balcony/measurements/{row["path"]}-{int(row["point"])}-a-URA-868/snapshots.html" target="_blank">{row["location"]} URA</a>')
    # do not show popup with BS
    if index == 0:
        popup = None

    folium.Marker(
        icon=folium.Icon(color=colors[row["path"]]),
        location=[row["latitude"], row["longitude"]],
        popup=popup
    ).add_to(m)

m.save("index.html")
