import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

root_dir = "D:\Stack\measurement-data"
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_cov_matrix(path):
    R = np.load(pjoin(path, "cov-matrix.npy"))
    return 10 * np.log10(np.abs(R))


def _plot(d):
    path = pjoin(root_dir, d)
    input = pjoin(path, "small-channel.npy")
    input_evm = pjoin(path, "raw-evm.txt")
    evm = np.loadtxt(input_evm, delimiter="\t")[:, 0]

    output = pjoin(path, "snapshots.html")

    # if os.path.isfile(output):
    #     return

    H_norm = np.load(input)
    # [snapshots x freq points x BS antennas x users]

    # remove last one bc faulty
    gain_50 = np.abs(H_norm[:, 0, 0:31])
    gain_51 = np.abs(H_norm[:, 1, 0:31])

    # if gain is 0 take the value of the other carrier freq.
    gain_50[gain_50 == 0] = gain_51[gain_50 == 0]
    gain_51[gain_51 == 0] = gain_50[gain_51 == 0]

    z = (20 * np.log10(gain_50) + 20 * np.log10(gain_51)) / 2

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{'type': 'surface', 'rowspan': 2}, {'type': 'xy'}], [None, {'type': 'xy'}]],
                        subplot_titles=["Channel gain", "EVM histogram", "Normalised covariance matrix"])

    fig.add_trace(go.Surface(z=z, cmin=z.min(), cmax=z.max(), showscale=True, colorbar=dict(x=0.45, y=0.5)), row=1,
                  col=1)
    # fig.update_xaxes(title_text="Antenna #", row=1, col=1)
    # fig.update_yaxes(title_text="Time", row=1, col=1)
    # fig.update_zaxes(title_text="Gain (dB)", row=1, col=1)

    dir_name = os.path.basename(os.path.dirname(output))
    fig.update_layout(
        title=f"{dir_name} Download norm. channel gain <a href = "
              f"\"https://dramco.be/projects/marrmot/balcony/measurements/{dir_name}/norm-channel.npy\">.npy</a> and "
              f"<a href = \"https://dramco.be/projects/marrmot/balcony/measurements/"
              f"{dir_name}/norm-channel.mat\">.mat</a>",
        scene=dict(
            xaxis_title="Antenna #",
            yaxis_title="Time",
            zaxis_title="Gain (dB)",
        ),
        font=dict(
            family="Courier New, monospace",
            size=14
        )
    )

    fig.add_trace(go.Histogram(x=evm), row=1, col=2)
    fig.update_layout(xaxis_title="EVM (%)",
                      yaxis_title="Count")

    R_norm = compute_cov_matrix(path)

    fig.add_trace(go.Heatmap(z=R_norm, colorscale='Viridis', colorbar=dict(xanchor="right", yanchor="top", len=0.5)),
                  row=2, col=2)

    fig['layout']['yaxis2'].update({'scaleanchor': "x2", 'scaleratio': 1})
    fig.write_html(output)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=5) as executor:
        for d in dirs:
            future = executor.submit(_plot, d)
            future.add_done_callback(lambda p: pbar.update())
