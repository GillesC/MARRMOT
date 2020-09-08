import os
from concurrent.futures import ProcessPoolExecutor
from os.path import join as pjoin

import numpy as np
from tqdm import tqdm

from utils.load_yaml import load_root_dir
from utils.util_loc import is_cont_meas, is_ula
import random

root_dir = load_root_dir()
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))


def compute_corr_coefficient(h1, h2):
    h1 = np.conjugate(h1.reshape(1, -1))
    h2 = h2.reshape(-1,1)

    return np.asscalar(np.abs(np.dot(h_herm, h)) / (np.linalg.norm(h1) * np.linalg.norm(h2)))



if __name__ == '__main__':
    H_ula = []
    H_ura = []

    for d in dirs:
        # [snapshots x freq points x BS antennas]
        path = os.path.join(root_dir, d, "small-channel.npy")
        H = np.load(input)
        if is_cont_meas(d):
        	# to much samples, split in 1 sec snapshots, i.e. 100 snapshots
        	H_splitted = np.array_split(H, H.shape[0] // 100)
        	for H_split in H_splitted:

		        if is_ula(d):
		            H_ula.append(H_split)
		        else:
		            H_ura.append(H_split)
		else:
			if is_ula(d):
		            H_ula.append(H)
		        else:
		            H_ura.append(H)

    antenna_idx = list(range(31))

   num_simulations = 10000

   results = {
	   "ULA": [[]*num_simulations]*31,
	   "URA": [[]*num_simulations]*31,
   }

    for sim in range(0,num_simulations):
	    for num_antennas in range(1, 32):
	        for conf, H_conf in zip(["ULA","URA"],[H_ula, H_ura]):
	            num_pos = len(H_conf)
	            pos_1 = random.randint(0, num_pos)
	            pos_2 = random.randint(0, num_pos)

	            # only two frequencies
	            freq = random.randint(0, 1)

	            random.shuffle(antenna_idx)
	            random_antennas = antenna_idx[:num_antennas]

	            h_1 = H_conf[pos_1]
	            h_2 = H_conf[pos_2]

	            # we have 1000 samples per meas
	            time_1 = random.randint(0, h_1.shape[0])
	            time_2 = random.randint(0, h_2.shape[0])

	            h_1= h_1[time_1, freq, random_antennas]
	            h_2= h_2[time_2, freq, random_antennas]

	            c = compute_corr_coefficient(h_1,h_2)

	            res[conf][num_antennas][sim] = c

	for conf, values in res.items():
		avg_c = np.mean(values, axis=1)
		plt.plot(avg_c, label=conf)

	plt.legend()
	plt.show()


