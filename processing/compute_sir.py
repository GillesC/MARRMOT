import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import extract_info_from_dir

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

max_simulations = 100


def sir(h: np.ndarray, h_others: np.ndarray):
    h = h.reshape(-1, 1)
    h_herm = np.conjugate(h).T

    interference = [np.abs(np.dot(h_herm, h_other.reshape(-1, 1))) ** 2 for h_other in h_others]

    interference = np.sum(interference)
    return np.asscalar(np.abs(np.dot(h_herm, h)) ** 2 / interference)


def avg_h(d):
    # [BS antennas]
    path = os.path.join(root_dir, d, "avg_H.npy")
    if os.path.isfile(path):
        H = np.load(path)

        path, point, num, conf, freq = extract_info_from_dir(d)

        return [path, point, conf, freq, H[:-1]]
    return None


if __name__ == '__main__':

    data = []
    for d in dirs:
        res = avg_h(d)
        if res is not None:
            data.append(res)

    df = pd.DataFrame(data, columns=['path', 'point', 'conf', 'freq', 'avg_h'])
    df = df.query("freq == 868")

    max_nodes = 10

    # TODO remove faulty last antenna

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    pbar = tqdm(total=2 * len(range(2, max_nodes)) * max_simulations)
    for name, group in df.groupby("conf"):
        group = group.reset_index()
        total_points = group.shape[0]
        points = np.arange(total_points)
        print(f"Examening {name} with {total_points} channel instances")
        x = []
        y = []
        for num_nodes in range(2, max_nodes):
            sum_rates = []
            for j in range(max_simulations):
                np.random.shuffle(points)
                nodes = points[:num_nodes]
                nodes_avg_h = group.iloc[nodes]['avg_h'].tolist()
                sum_rate = 0
                for i in range(num_nodes):
                    h_others = nodes_avg_h[:i]
                    h_others.extend(nodes_avg_h[i + 1:])
                    h = np.array(nodes_avg_h[i])
                    sum_rate += np.log2(1 + sir(h, np.array(h_others)))
                sum_rates.append(sum_rate)
                pbar.update()
            avg_sum_rate = np.mean(sum_rates)
            y.append(avg_sum_rate)
            x.append(num_nodes)

        ax.plot(x, y, label=name)
    ax.legend()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    pbar = tqdm(total=2 * len(range(2, max_nodes)) * max_simulations)
    for name_conf, group_conf in df.groupby("conf"):
        for name_conf_path, group_conf_path in group_conf.groupby("path"):

            if name_conf_path == "D":
                continue

            group_conf_path = group_conf_path.reset_index()
            total_points = group_conf_path.shape[0]
            max_nodes = total_points if total_points < max_nodes else max_nodes

            points = np.arange(total_points)
            print(f"Examining {name_conf_path} with {total_points} channel instances")
            x = []
            y = []
            for num_nodes in range(2, max_nodes):
                sum_rates = []
                for j in range(max_simulations):
                    np.random.shuffle(points)
                    nodes = points[:num_nodes]
                    nodes_avg_h = group_conf_path.iloc[nodes]['avg_h'].tolist()
                    sum_rate = 0
                    for i in range(num_nodes):
                        h_others = nodes_avg_h[:i]
                        h_others.extend(nodes_avg_h[i + 1:])
                        h = np.array(nodes_avg_h[i])
                        sum_rate += np.log2(1 + sir(h, np.array(h_others)))
                    sum_rates.append(sum_rate)
                    pbar.update()
                avg_sum_rate = np.mean(sum_rates)
                y.append(avg_sum_rate)
                x.append(num_nodes)
            ax.plot(x, y, label=name_conf+"-"+name_conf_path)
    ax.legend()
    plt.show()
