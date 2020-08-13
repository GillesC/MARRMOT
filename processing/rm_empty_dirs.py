import os
import shutil

root_dir = os.path.dirname(os.path.abspath(__file__))
subdir, dirs, files = next(os.walk(os.path.join(root_dir, "measurements")))

for dir in dirs:
    dir_path = os.path.join(root_dir, "measurements", dir)
    if len(os.listdir(dir_path)) == 0: # Check is empty..
        shutil.rmtree(dir_path) # Delete..