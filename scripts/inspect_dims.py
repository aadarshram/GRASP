import h5py
import os

data_dir = 'data/metaworld_task'
files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
if files:
    with h5py.File(os.path.join(data_dir, files[0]), 'r') as f:
        print("Action shape:", f['action'].shape)
        print("Qpos shape:", f['observations/qpos'].shape)
else:
    print("No files found")
