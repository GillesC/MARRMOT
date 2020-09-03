def extract_info_from_dir(d):
    import os
    if os.path.exists(os.path.dirname(d)):
        d = os.path.basename(d)
    args = tuple(d.split("-"))

    if args[1] == 0:
        path, point, num, conf, freq = args
    else:
        path, point, conf, freq = args
        num = 'a'
    return path, int(point), num, conf, int(freq)


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

