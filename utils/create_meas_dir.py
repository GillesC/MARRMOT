paths = ["A", "B", "C", "D"]
points = [14, 14, 11, 3]
confs = ["ULA", "URA"]
freqs = ["2610"]

import os
from os.path import join as pjoin

root_dir = "D:\Stack\measurement-data"
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

if __name__ == '__main__':
    for path, point in zip(paths, points):
        for p in range(1, 1 + point):
            for conf in confs:
                for freq in freqs:
                    dir_path = pjoin(root_dir, f"{path}-{p}-a-{conf}-{freq}")
                    if not os.path.isdir(dir_path):
                        os.mkdir(dir_path)
