paths = ["A", "B", "C", "D"]
points = [14, 16, 14, 3]
confs = ["ULA", "URA"]
freqs = ["868"]

import os
from os.path import join as pjoin

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))

if __name__ == '__main__':
    for path, point in zip(paths, points):
        for p in range(1, 1 + point):
            for conf in confs:
                for freq in freqs:
                    dir_path = pjoin(root_dir, "measurements", f"{path}-{p}-a-{conf}-{freq}")
                    if not os.path.isdir(dir_path):
                        os.mkdir(dir_path)
