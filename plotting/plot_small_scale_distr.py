import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from processing import distributions as dist

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def _plot(d):
    path = pjoin(root_dir, d)
    input = pjoin(path, "small-channel.npy")

    output = pjoin(path, "small-scale-distr")

    H = np.load(input)
    # [snapshots x freq points x BS antennas x users]

    # remove last one bc faulty
    r = np.abs(H[:, :, 0:31]).flatten()
    plt.cla()

    best_distribution, best_params, best_sse, (y, x) = dist.best_fit_distribution(r, ax=None,
                                                                                  verbose=False)
    pdf = dist.make_pdf(best_distribution, best_params)

    K = -1
    if best_distribution.name == "rice":
        mu_p = np.abs(np.mean(r))
        sigma_p = np.std(r)

        K = np.sqrt(mu_p ** 2 - sigma_p ** 2) / (mu_p - np.sqrt(mu_p ** 2 - sigma_p ** 2))
        K = 10 * np.log10(K)

    ax = pdf.plot(lw=2,
                  label=f'PDF best {best_distribution.name} K={K}dB',
                  legend=True)
    sns.distplot(r, ax=ax)

    # fig = ff.create_distplot([r], ["abs(h)"])

    plt.savefig(output + ".png")

    # fig.write_html(output + ".html")


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=5) as executor:
        for d in dirs:
            future = executor.submit(_plot, d)
            future.add_done_callback(lambda p: pbar.update())
