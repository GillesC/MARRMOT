import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio

root_dir = "D:\Stack\measurement-data"
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compress_channel(d):


    path = os.path.join(root_dir, d, "channel.npy")
    # [snapshots x freq points x BS antennas x users]
    H = np.load(path)
    h_compressed = H[:, 50:52, :, 0]

    if not os.path.isfile(os.path.join(root_dir, d, "small-channel.npy")):
        np.save(os.path.join(root_dir, d, "small-channel.npy"), h_compressed)

    if not os.path.isfile(os.path.join(root_dir, d, "channel.mat")):
        H_mat = {'H': h_compressed}
        sio.savemat(os.path.join(root_dir, d, "channel.mat"), H_mat)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=3) as executor:
        for d in dirs:
            future = executor.submit(compress_channel, d)
            future.add_done_callback(lambda p: pbar.update())
