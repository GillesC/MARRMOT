# Renaming old filename to new filenames
import os
from os.path import join as pjoin

root_dir = "D:\Stack\measurement-data"
subdir, dirs, files = next(os.walk(os.path.join(root_dir)))

if __name__ == '__main__':
    for d in dirs:
        path = os.path.join(root_dir, d)
        subdir, dirs, files = next(os.walk(path))
        if "channel.mat" in files:
            # os.remove(pjoin(path, "channel.mat"))
            continue
        for f in files:
            file_path = os.path.join(root_dir, d, f)
            file_path_new = os.path.join(root_dir, d)
            if "Outdoor" in f:
                if "EVM" in f:
                    os.rename(file_path, pjoin(file_path_new, "raw-evm.txt"))
                else:
                    os.rename(file_path, pjoin(file_path_new, "raw-channel.txt"))



