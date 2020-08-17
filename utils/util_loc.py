def extract_info_from_dir(d):
    import os
    if os.path.exists(os.path.dirname(d)):
        d = os.path.basename(d)
    path, point, num, conf, freq = tuple(d.split("-"))
    return path, int(point), num, conf, freq


def get_meas(path, point):
    "Ignores 0 points as they are continuous measurements"
    if int(point) == 0:
        return None

    from os.path import join as pjoin
    import pandas as pd
    import os

    root_dir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(pjoin(root_dir, "..", "gps-loc.csv"))
    res = df.query(f"path == \"{path}\" and point == \'{point}\'")
    return res.iloc[0]

