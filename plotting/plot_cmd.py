import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm
import utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))

df_list = []

if __name__ == '__main__':
    for d in dirs:
        d_cmd = np.load(pjoin(root_dir, "measurements", d, "d_cmd.npy"))
        path, point, num, conf, freq = utils.extract_info_from_dir(d)
        res = utils.get_meas(path, point)
        if res is None:
            continue
        dist = res["distance"]
        if dist is np.nan:
            continue
        df_list.append({
            "distance": float(dist),
            "path": str(path),
            "cmd": float(d_cmd),
            "conf": str(conf)
        })

df = pd.DataFrame(data=df_list)

# keep only non-D path rows
df = df[df['path'] != "D"]



for (path_name, conf_name), df_group in df.sort_values(by=["path", 'distance']).groupby(["path", "conf"]):
    x = df_group["distance"]
    y = df_group["cmd"]
    linestyle = '-'
    if conf_name == "URA":
        linestyle = '--'

    plt.plot(x, y, linestyle=linestyle, linewidth=2, label=f"{path_name}-{conf_name}")

plt.legend()
plt.show()

# for conf_name, df_group in df.groupby("conf"):
#     for path_name, path_group in df_group.groupby("path"):
#         x = path_group["distance"]
#         y = path_group["cmd"]
#         zipped = zip(x, y)
#         x, y = sorted(zipped, key=lambda t: t[0])
#         plt.plot(x,y)

# sns.scatterplot(x="distance", y="cmd", hue="path", data=df.query("conf == \"ULA\""))
# plt.show()
#
# sns.scatterplot(x="distance", y="cmd", hue="path", data=df.query("conf == \"URA\""))
# plt.show()
