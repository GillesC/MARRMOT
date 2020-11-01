import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = load_root_dir()
    subdir, dirs, files = next(os.walk(os.path.join(root_dir)))
    for d in dirs:
        # only process 868 measurements
        is_868_b = is_868(d)
        is_ula_b = is_ula(d)

        if not is_868_b:
            print(f"Skipped {d}")
            continue
        print(f"processing {d}")
        H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
        # remove faulty antenna 32
        # [snapshots x freq points x BS antennas]
        H = H[:, :, :]
        H = np.nanmean(H, axis=1)
        plt.cla()
        if is_cont_meas(d):
            path = get_path(d)

            conf = "ULA" if is_ula(d) else "URA"



            for M in [1, 32]:
                if M == 1 and conf == "ULA":
                    gain = 10 * np.log10(np.abs(H[:, 8]) ** 2)
                else:
                    gain = 10 * np.log10(np.sum(np.abs(H[:, :M]) ** 2, axis=1))
                # if M==1:
                plt.plot(gain, label=f"{M}", linewidth=1, alpha=0.6) #, color="white")
                # else:
                #     plt.plot(gain, label=f"{M}", linewidth=1, alpha=0.95, color="C1")

            # plt.legend()
            plt.title(conf + "-" + path)
            axes = plt.gca()
            # axes.spines["bottom"].set_color("white")
            # axes.spines["left"].set_color("white")
            # axes.xaxis.label.set_color('white')
            # axes.tick_params(colors='white')
            plt.savefig(f"chhard-{conf}-{path}.png", transparent=True)
