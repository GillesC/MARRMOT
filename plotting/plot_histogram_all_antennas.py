import os
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import distributions as dist
from utils.load_yaml import *

root_dir = abspath(pjoin(dirname(__file__), ".."))

conf = get_conf()

for env in conf["environments"]:
    env_path = abspath(pjoin(root_dir, "data", env))
    if exists(env_path):
        print(f"{env}")

        all_signal_ampl = []

        for scenario in filter(lambda x: os.path.isdir(pjoin(env_path, x)), os.listdir(env_path)):
            print(f"\t{scenario}")
            ax = None

            scenario_dir_path = pjoin(env_path, scenario)
            input_channels_path = pjoin(scenario_dir_path, "channels_data_carr_all_positions.ftr")
            df_channels = pd.read_feather(input_channels_path)

            df_channels = df_channels.query("(SubCarrier == 51 or SubCarrier == 52) and User == 0")

            signal_ampl = df_channels["ChannelAmplitude"].dropna()
            all_signal_ampl.append(signal_ampl)

            plt.figure(figsize=(12, 8))
            ax = sns.distplot(signal_ampl)
            ax.set_title(f'Channel Amplitude for scenario {scenario}')
            ax.set_xlabel(u'Channel Amplitude')
            ax.set_ylabel('Frequency')

            # (scale, b, loc), bins = dist.fit(data=powers, dist=st.rice)
            # pdf = dist.make_pdf(st.rice, (b, loc, scale))
            # ax = pdf.plot(lw=2, label=f'PDF', legend=True)

            #
            # k = dist.get_k_factor(params)
            # ax = pdf.plot(lw=2, label=f'PDF (K={k})', legend=True)
            #
            # # plt.legend()
            # plt.show()

            best_distribution, best_params, best_sse, (y, x) = dist.best_fit_distribution(signal_ampl, ax=ax,
                                                                                          verbose=True)

            sigma = best_params[-1]
            b = best_params[0]
            v = b * sigma

            k = 10 * np.log10(v ** 2 / (2 * sigma ** 2))

            pdf = dist.make_pdf(best_distribution, best_params)
            omega = r'$\Omega$'
            ax = pdf.plot(lw=2,
                          label=f'PDF best {best_distribution.name} with K= {k:.2f}dB {omega}={10 * np.log10(v ** 2 + 2 * sigma ** 2):.2f}dB',
                          legend=True)
            plt.legend()
            plt.savefig(pjoin(env_path, scenario, "distr.png"), transparent=True)
            plt.show()

all_ampl = pd.concat(all_signal_ampl)
plt.figure(figsize=(12, 8))
ax = sns.distplot(all_ampl)
ax.set_title(f'Channel Amplitude for all scenario')
ax.set_xlabel(u'Channel Amplitude')
ax.set_ylabel('Frequency')
best_distribution, best_params, best_sse, (y, x) = dist.best_fit_distribution(all_ampl, ax=ax, verbose=True)
sigma = best_params[-1]
b = best_params[0]
v = b * sigma

k = 10 * np.log10(v ** 2 / (2 * sigma ** 2))

pdf = dist.make_pdf(best_distribution, best_params)
omega = r'$\Omega$'
ax = pdf.plot(lw=2,
              label=f'PDF best {best_distribution.name} with K= {k:.2f}dB {omega}={10 * np.log10(v ** 2 + 2 * sigma ** 2):.2f}dB',
              legend=True)
plt.legend()
plt.show()
