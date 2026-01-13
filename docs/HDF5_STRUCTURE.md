# HDF5 File Structure Documentation

## Overview
This document describes the complete structure of HDF5 files created by the GRASP data generation pipeline. Each episode is stored as an HDF5 file containing robot trajectories, multi-camera observations, actions, joint states, and task descriptions.

## Complete HDF5 Hierarchy

```
episode_XXXX.hdf5
├── /action                              (Dataset)
│   └── Shape: (T, action_dim)
│       Data Type: float32
│       Description: Robot action commands at each timestep
│       Dimensions:
│       ├── T: Episode length (number of timesteps)
│       └── action_dim: Action dimensionality (typically 7-9 for robot arms)
│       Stats: Min, Max, Mean, Std values across all timesteps
│
├── /observations                        (Group)
│   │
│   ├── qpos                            (Dataset)
│   │   └── Shape: (T, state_dim)
│   │       Data Type: float32
│   │       Description: Joint positions (degrees or radians) at each timestep
│   │       Dimensions:
│   │       ├── T: Episode length
│   │       └── state_dim: Degrees of freedom (typically 7 for 7-DOF robot)
│   │       Stats: Min, Max, Mean, Std for all DOFs
│   │
│   ├── qvel                            (Dataset)
│   │   └── Shape: (T, state_dim)
│   │       Data Type: float32
│   │       Description: Joint velocities (rad/s) at each timestep
│   │       Dimensions:
│   │       ├── T: Episode length
│   │       └── state_dim: Degrees of freedom
│   │       Stats: Min, Max, Mean, Std for all DOFs
│   │
│   └── images                          (Group)
│       │
│       ├── top                         (Dataset)
│       │   └── Shape: (T, H, W, 3)
│       │       Data Type: uint8
│       │       Compression: gzip (optional)
│       │       Description: RGB images from top camera view
│       │       Dimensions:
│       │       ├── T: Episode length
│       │       ├── H: Image height (typically 128 or 256 pixels)
│       │       ├── W: Image width (typically 128 or 256 pixels)
│       │       └── 3: RGB channels (Red, Green, Blue)
│       │       Uncompressed Size: ~T × H × W × 3 bytes
│       │
│       ├── left                        (Dataset) [if selected]
│       │   └── Same structure as 'top'
│       │       Description: RGB images from left camera view
│       │
│       ├── right                       (Dataset) [if selected]
│       │   └── Same structure as 'top'
│       │       Description: RGB images from right camera view
│       │
│       ├── front                       (Dataset) [if selected]
│       │   └── Same structure as 'top'
│       │       Description: RGB images from front camera view
│       │
│       └── gripper                     (Dataset) [if selected]
│           └── Same structure as 'top'
│               Description: RGB images from gripper-mounted camera
│
├── /language_raw                        (Dataset)
│   └── Shape: (1,)
│       Data Type: string (variable-length UTF-8)
│       Description: Natural language description of the task/instruction
│       Example: "Pick up the red cube and place it in the box"
│
└── Attributes (Root Level)
    └── sim: Boolean (True for simulation data, False for real robot)
        Indicates whether the episode comes from simulation (MetaWorld/MuJoCo)
        or real-world robot execution
```

## Detailed Field Descriptions

### /action Dataset
- **Purpose**: Stores the control commands sent to the robot actuators
- **Shape**: `(T, action_dim)` where T is episode length
- **Values**: Normalized or raw action values depending on environment
- **MetaWorld**: Typically 7-9D (e.g., 7D for gripper position + 1-2D for gripper state)
- **Access**: `episode_data['action'][t, :]` → action at timestep t

### /observations/qpos Dataset
- **Purpose**: Joint positions representing robot configuration at each moment
- **Shape**: `(T, state_dim)` where state_dim matches robot DOF
- **Units**: Radians for joint angles (some environments use degrees)
- **MetaWorld Arm**: Typically 7-DOF (7 joint angles) + gripper state
- **Access**: `episode_data['observations/qpos'][t, :]` → joint positions at t
- **Usage**: Represents full robot state for trajectory learning

### /observations/qvel Dataset
- **Purpose**: Joint velocities showing rate of change of joint angles
- **Shape**: `(T, state_dim)` matching qpos dimensions
- **Units**: Radians per second (rad/s)
- **Access**: `episode_data['observations/qvel'][t, :]` → joint velocities at t
- **Usage**: Important for dynamics models and momentum-based methods

### /observations/images/{camera_name} Dataset
- **Purpose**: Store RGB images from each selected camera
- **Cameras Available**:
  - `top`: Bird's eye view of the workspace
  - `left`: Left-side view of the robot and task
  - `right`: Right-side view
  - `front`: Front view of the workspace
  - `gripper`: Egocentric view from gripper-mounted camera
- **Shape**: `(T, H, W, 3)` for each camera
  - T: Episode length
  - H: Height (typically 128 or 256)
  - W: Width (typically 128 or 256)
  - 3: RGB channels
- **Format**: RGB uint8 (values 0-255)
- **Compression**: gzip compression applied (optional, preserves lossless quality)
- **Access**: `episode_data['observations/images/top'][t, :, :, :]` → single frame
- **Dynamic Selection**: Only selected cameras (from `camera_config.py`) are stored

### /language_raw Dataset
- **Purpose**: Natural language instruction describing the task
- **Shape**: `(1,)` - Single string value
- **Type**: UTF-8 variable-length string
- **Format**: Human-readable task description
- **Examples**:
  - "Pick up the red cube"
  - "Open the drawer"
  - "Press the button three times"
- **Access**: `episode_data['language_raw'][0]` → task description
- **Encoding**: Decoded from bytes using UTF-8

### sim Attribute
- **Purpose**: Flag indicating data source
- **Value**: `True` for simulated data, `False` for real robot
- **Access**: `f.attrs['sim']` → Boolean value
- **Usage**: Distinguish synthetic vs. real-world episodes

## Camera Configuration

The cameras stored in each HDF5 file are determined by `camera_config.py`:

```python
from camera_config import SELECTED_CAMERAS, get_camera_names()

# Get list of cameras to expect in HDF5 files
cameras = get_camera_names()  # Returns: ['top', 'left', 'right'] by default
```

### MetaWorld Camera Details

| Camera | ID | Position | View | Typical Use |
|--------|----|---------:|------|------------|
| top | 0 | Above workspace | Bird's eye | Global context |
| left | 1 | Left of robot | Side view | Manipulation visibility |
| right | 2 | Right of robot | Side view | Complement to left |
| front | 4 | Front of robot | Frontal view | Precise hand-object interaction |
| gripper | 5 | Gripper tip | Egocentric | Fine-grained tactile tasks |

## File Statistics

### Typical File Size
```
Single Episode with 3 cameras (T=100):
├── actions: 100 × 7 × 4 bytes = 2.8 KB
├── qpos: 100 × 7 × 4 bytes = 2.8 KB
├── qvel: 100 × 7 × 4 bytes = 2.8 KB
└── images: 3 × 100 × 128 × 128 × 3 × 1 byte (gzip ~50-70% reduction)
   ├── Uncompressed: ~150 MB
   └── Compressed: ~45-75 MB (depending on scene complexity)

Total per episode: ~45-75 MB (with gzip compression)
```

### Compression Details
- **Format**: gzip (lossless compression)
- **Compression Ratio**: ~50-70% typical (depends on image complexity)
- **Access**: Transparent to user - h5py handles decompression automatically
- **Tradeoff**: Smaller files, slower read speed

## Usage Examples

### Reading Complete Episode

```python
import h5py
import numpy as np
from camera_config import get_camera_names

# Open file
with h5py.File('episode_0.hdf5', 'r') as f:
    # Get metadata
    is_sim = f.attrs['sim']
    
    # Get language instruction
    task = f['/language_raw'][0].decode('utf-8')
    
    # Get trajectory length
    T = f['/action'].shape[0]
    
    # Get action sequence (T, 7)
    actions = f['/action'][:]
    
    # Get joint positions over time (T, 7)
    qpos = f['/observations/qpos'][:]
    
    # Get joint velocities over time (T, 7)
    qvel = f['/observations/qvel'][:]
    
    # Get images from each camera
    cameras = get_camera_names()
    for t in range(T):
        for cam in cameras:
            img = f[f'/observations/images/{cam}'][t]  # Shape: (H, W, 3)
            # Process image...
```

### Inspecting File Structure

```python
import h5py

with h5py.File('episode_0.hdf5', 'r') as f:
    # Print all datasets and groups
    def print_structure(name, obj):
        print(name)
        if isinstance(obj, h5py.Dataset):
            print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
    
    f.visititems(print_structure)
    
    # Print attributes
    for attr_name, attr_value in f.attrs.items():
        print(f"{attr_name}: {attr_value}")
```

### Analyzing Episode Statistics

```python
import h5py
import numpy as np

with h5py.File('episode_0.hdf5', 'r') as f:
    action = f['/action'][:]
    
    print(f"Action sequence statistics:")
    print(f"  Shape: {action.shape}")
    print(f"  Min: {np.min(action):.3f}")
    print(f"  Max: {np.max(action):.3f}")
    print(f"  Mean: {np.mean(action):.3f}")
    print(f"  Std: {np.std(action):.3f}")
```

## Data Generation Parameters

When generating episodes with `generate_metaworld_data.py`:

```bash
python scripts/generate_metaworld_data.py \
    --task_name pick-place-v3 \
    --episode 10 \
    --output_dir data/metaworld_dataset/ \
    --save_video  # Optional: saves MP4 in addition to HDF5
```

### What Gets Created

```
data/metaworld_dataset/
├── episode_0.hdf5        # HDF5 with trajectory data
├── episode_1.hdf5
├── ...
└── videos/               # Optional (if --save_video flag used)
    ├── episode_0.mp4
    ├── episode_1.mp4
    └── ...
```

## Visualization and Inspection

Use the enhanced `visualize_dataset.py` to inspect all HDF5 contents:

```bash
# Print complete structure and statistics of an episode
python scripts/visualize_dataset.py --file data/episode_0.hdf5 --info-only

# Visualize video with overlaid state information
python scripts/visualize_dataset.py --file data/episode_0.hdf5

# Batch inspect all episodes in directory
python scripts/visualize_dataset.py --dir data/metaworld_dataset/
```

### Output Includes
- File metadata (size, source, compression)
- Task description
- Action statistics and ranges
- Joint position/velocity statistics
- Camera information (resolution, compression, size)
- Episode length and timing information

## Notes

1. **Camera Selection**: Only cameras in `camera_config.SELECTED_CAMERAS` are stored in new HDF5 files
2. **Backward Compatibility**: Old files may contain different cameras; visualization adapts dynamically
3. **Compression**: Image compression is lossless (gzip) - no quality loss
4. **Video Files**: If `--save_video` flag is used during generation, MP4s are saved separately (not in HDF5)
5. **Memory Efficient**: h5py allows lazy loading - only load the data you need
6. **Timestamps**: Currently no explicit timestamp data; assume uniform sampling at control frequency (typically 10-50 Hz)

## See Also
- [camera_config.py](../scripts/camera_config.py) - Camera selection configuration
- [generate_metaworld_data.py](../scripts/generate_metaworld_data.py) - Data generation script
- [visualize_dataset.py](../scripts/visualize_dataset.py) - Visualization and inspection tool
