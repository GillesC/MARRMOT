import numpy as np


def pd_to_heatmap_matrix(df, x=None, y=None):
    if x is None:
        x = df["XPos"].unique()
        x.sort()
    if y is None:
        y = df["YPos"].unique()
        y.sort()

    X, Y = np.meshgrid(x, y)

    Z = np.zeros(shape=(len(x), len(y)))

    for x_idx, x_pos in enumerate(x):
        for y_idx, y_pos in enumerate(y):
            power = df.query(f"XPos == {x_pos} and YPos == {y_pos}")[
                "ChannelPower_dB"].mean()
            Z[x_idx, y_idx] = power

    return Z, (X,Y)
