def extract_info_from_dir(d):
    import os
    if os.path.exists(os.path.dirname(d)):
        d = os.path.basename(d)
    args = tuple(d.split("-"))

    if int(args[1]) == 0:
        path, point, conf, freq = args
        num = 'a'
    else:
        path, point, num, conf, freq = args
    return path, int(point), num, conf, int(freq)


def is_868(d):
    _, _, _, _, freq = extract_info_from_dir(d)
    return freq == 868

def get_path(d):
    path, _, _, _, _ = extract_info_from_dir(d)
    return path

def get_point(d):
    _, point, _, _, _ = extract_info_from_dir(d)
    return int(point)

def is_ula(d):
    _, _, _, conf, _ = extract_info_from_dir(d)
    return conf == "ULA"


def is_cont_meas(d):
    _, point, _, _, _ = extract_info_from_dir(d)

    if point == 0:
        return True
    else:
        return False


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
