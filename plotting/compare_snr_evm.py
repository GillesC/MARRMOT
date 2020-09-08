from os.path import join as pjoin

import pandas as pd

from utils.load_yaml import load_root_dir
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import LatexifyMatplotlib as lm

root_dir = load_root_dir()

if __name__ == '__main__':
    df = pd.read_hdf(pjoin(root_dir, "db.h5"), key="df")

    # drop unnec. columns
    del df['H']
    del df['H_avg']

    # only interested in static 868 meas
    df = df.query("freq == 868")
    df = df.query("is_cont == False")
    df = df.query("num == \'a\'")

    df = df.sort_values(by=['path', 'point'])
    df["loc"] = df["path"] + df["point"].astype(str)

    locs = df["loc"].unique()

    paths = df["path"].unique()

    snr_ula = dict.fromkeys(paths, [])
    snr_ura = dict.fromkeys(paths, [])

    for loc in locs:
        ula = df.query(f"loc == \'{loc}\' and conf == \"ULA\"")
        ura = df.query(f"loc == \'{loc}\' and conf == \"URA\"")

        if not ula.empty and not ura.empty:
            ula_snr = ula["snr"].values[0].flatten().tolist()
            ura_snr = ura["snr"].values[0].flatten().tolist()

            arr = snr_ula.get(loc[0]).copy()
            arr.extend(ula_snr)
            snr_ula[loc[0]] = arr

            arr = snr_ura.get(loc[0]).copy()
            arr.extend(ura_snr)
            snr_ura[loc[0]] = arr


        else:
            print(f"ULA/URA not avail for loc {loc}")

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    colors = dict(zip(paths, colors[:len(paths)]))

    for conf, snr_uxa in zip(["ULA", "URA"], [snr_ula, snr_ura]):
        for key, values in snr_uxa.items():
            ecdf = ECDF(values)
            linestyle = '--' if conf == "URA" else "-"
            _ = plt.plot(ecdf.x, ecdf.y, linestyle=linestyle, lw=2, label=f"{conf} - {key}", color=colors[key])

    # Label axes and show plot
    _ = plt.xlabel('SNR', size=14)
    _ = plt.ylabel('ECDF', size=14)
    plt.legend(ncol=2)
    lm.save("cdf-snr.tex", scale_legend=0.7, show=True, plt=plt)


