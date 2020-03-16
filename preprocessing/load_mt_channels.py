import scipy.io as sio
from os.path import dirname, join as pjoin


root_dir = dirname(__file__)
data_dir = pjoin(root_dir, "..", "data")
channels_file = pjoin(data_dir, "H.mat")

py_channels = sio.loadmat(channels_file)
print(py_channels)