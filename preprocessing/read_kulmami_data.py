"""
%% Script to read data from KulMaMi
% Reads a .txt file created during KulMaMi-measurements and saves the
% channel matrix as a .npy-file
%
% 1) Iterate over the dirs in the root dir
% 2) Remove base station antennas not present in the measurements
% 3) Recombine Re- and Im-parts
% 4) Create H as [snapshots x freq points x BS antennas x users]
% 5) Rearrange M and K dimensions
% 6) If activated, remove outliers (snapshots lost, possibly due to lost sync)
% 7) Sort the base station antennas as in the physical order
% 8) If needed, remove PN sequence (old measurements)
% 9) Save the processed channel matrix in a .mat-file, or separate files
% per UE
"""

import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm


def load_config():
    import yaml
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "config.yml")) as file:
        return yaml.full_load(file)


cfg = load_config()
root_dir = cfg["root_dir"]
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))
K = cfg["num_layers"]
M = cfg["num_bs_antennas"]
F = cfg["num_rsrc_blocks"]
block_size = cfg["block_size"]
num_subcarriers = cfg["num_subcarriers"]


def process_channel(d):
    path = os.path.join(root_dir, d)
    input = pjoin(path, "raw-channel.txt")
    output = pjoin(path, "channel")
    if os.path.isfile(output + ".npy"):
        return

    print(d, flush=True)
    arr = np.loadtxt(input)

    remainder = arr.shape[0] % (2 * F * K)
    # calculate number of snapshots
    N = arr.shape[0] // (2 * F * K)

    if remainder != 0:
        print(f"We have some 'brol' discarding {remainder} lines, keep {N} snapshots")
        arr = arr[:N*(2 * F * K)]


    # Relevant subcarriers (100/user)
    subcarriers = F * K

    assert arr.shape[1] == M, "Number of recorded antennas is not the same as the num configured antennas"
    assert arr.shape[0] % 2 == 0, "Number of channel values should be divisible by two (Re+Im)"

    # merge Im and Re in array
    arr_complex = np.zeros(shape=(arr.shape[0] // 2, M), dtype=complex)

    for s in range(N):
        start_re = s * 2 * subcarriers
        end_re = s * 2 * subcarriers + subcarriers
        end_im = (s + 1) * 2 * subcarriers
        arr_complex[s * subcarriers:(s + 1) * subcarriers, :] = arr[start_re:end_re] + 1j * arr[end_re: end_im]

    del arr

    # create channel matrix
    # [snapshots x freq points x BS antennas x users]
    # M and K needs to descrambled later on

    # @njit(parallel=True)
    def create_temp_channel(arr_complex: np.ndarray, N: int, F: int, M: int, K: int, subcarriers: int) -> np.ndarray:
        H_temp = np.zeros(shape=(N, F, M, K), dtype=complex)
        for n in range(N):
            for f in range(F):
                for k in range(K):
                    H_temp[n, f, :, k] = arr_complex[n * subcarriers + k + f * K, :]
        return H_temp

    H_temp = create_temp_channel(arr_complex, N, F, M, K, subcarriers)

    del arr_complex

    # descramble M and K
    H = np.zeros_like(a=H_temp)
    a = np.array([0, 4, 8, 12, 16, 20, 24, 28])
    ant = np.concatenate([np.tile(a, 4), np.tile(a + 1, 4), np.tile(a + 2, 4), np.tile(a + 3, 4)])

    idx = 0
    for k in range(K):
        for m in range(M):
            if k < 4:
                u = m // 8
            elif k < 8:
                u = 4 + m // 8
            else:
                u = 8 + m // 8
            H[:, :, m, k] = H_temp[:, :, ant[idx], u]
            # reset index
            if k == 3 or k == 7:
                idx = 0
            else:
                idx += 1
    del H_temp

    np.save(output + ".npy", H)
    # to save space on the disk only sa ve .npy file
    # H_mat = {'H': H}
    # sio.savemat(output + ".mat", H_mat)


if __name__ == '__main__':
    pbar = tqdm(total=len(dirs))
    with ProcessPoolExecutor(max_workers=10) as executor:
        for d in dirs:
            future = executor.submit(process_channel, d)
            future.add_done_callback(lambda p: pbar.update())
