import os
import random

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula, is_868, get_path, get_point
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = load_root_dir()
    subdir, dirs, files = next(os.walk(os.path.join(root_dir)))
    ax = plt.gca()

    pbar = tqdm(total=len(dirs))

    URA = "URA"
    ULA = "ULA"

    res = {
        URA: np.zeros(32),
        ULA: np.zeros(32),
        "A": np.zeros(32),
        "B": np.zeros(32),
        "C": np.zeros(32),
    }

    cnt = {
        URA: 0,
        ULA: 0,
    }

    for d in dirs:
        # only process 868 measurements
        is_868_b = is_868(d)
        is_ula_b = is_ula(d)

        if not is_868_b:
            # print(f"Skipped {d}")
            continue
        # print(f"processing {d}")
        H = np.load(os.path.join(root_dir, d, "small-channel.npy"))
        # remove faulty antenna 32
        # [snapshots x freq points x BS antennas]
        # H = np.mean(H, axis=(0, 1))

        conf = ULA if is_ula(d) else URA

        # gets an 1xM antenna
        tmp = np.zeros(H.shape[2])
        tmp_cnt = 0
        # for s in H:
        #     for f in s:
        #         tmp_cnt += 1
        #         for m in range(f.shape[0]):
        #             tmp[m] += np.linalg.norm(H[:m + 1]) ** 2
        for s in H:
            f = s[0, :]
            tmp_cnt += 1
            for m in range(f.shape[0]):
                tmp[m] += np.linalg.norm(H[:m + 1]) ** 2
        # divide by 1 antenna to get the gain ratio wrt one antenna
        tmp /= tmp_cnt
        tmp /= tmp[0]
        res[conf] += tmp
        cnt[conf] += 1

        if is_cont_meas(d):
            path = get_path(d)
            if not is_ula_b:
                path = path + "R"
            res[path] = tmp
        pbar.update()

    res[URA] = res[URA] / cnt[URA]
    res[ULA] = res[ULA] / cnt[ULA]

    ura_log = 10 * np.log10(res[URA])
    ula_log = 10 * np.log10(res[ULA])

    plt.plot(np.arange(1, 33), ura_log, label=f"URA", linewidth=1)
    plt.plot(np.arange(1, 33), ula_log, label=f"ULA", linewidth=1)
    plt.plot(np.arange(1, 33), 20 * np.log10(np.arange(1, 33)), label=f"M^2 dB", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["A"]), label=f"A (ULA)", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["B"]), label=f"B (ULA)", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["C"]), label=f"C (ULA-", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["AR"]), label=f"A (URA)", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["BR"]), label=f"B (URA)", linewidth=1)
    plt.plot(np.arange(1, 33), 10 * np.log10(res["CR"]), label=f"C (URA-", linewidth=1)
    plt.legend()
    plt.show()
    from plotting import LatexifyMatplotlib as lm

    # lm.save("gain_per_antenna.tex", scale_legend=0.7, show=True, plt=plt)
