import glob
import os
from concurrent.futures import ThreadPoolExecutor
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import scipy.io as sio


def load_config():
    import yaml
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config.yml")) as file:
        return yaml.full_load(file)


cfg = load_config()
root_dir = cfg["root_dir"]
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compress_channel(input_file):
    output = input_file.replace("raw", "small")
    if os.path.isfile(output):
        return

    if os.path.isfile(input_file):
        # [snapshots x freq points x BS antennas x users]
        H = np.load(input_file)
        h_compressed = H[:, 50:52, :, 1]

        #if not os.path.isfile(os.path.join(root_dir, d, "small-channel.npy")):
        np.save(output, h_compressed)

        #if not os.path.isfile(os.path.join(root_dir, d, "channel.mat")):
        #    H_mat = {'H': h_compressed}
        #    sio.savemat(os.path.join(root_dir, d, "channel.mat"), H_mat)


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            measurements = glob.glob(os.path.join(root_dir, d, 'raw-channel-*.npy'))
            pbar = tqdm(total=len(measurements))
            for meas in measurements:
                future = executor.submit(compress_channel, meas)
                future.add_done_callback(lambda p: pbar.update())
            pbar.close()


