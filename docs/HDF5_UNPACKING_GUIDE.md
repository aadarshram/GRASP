# HDF5 Data Unpacking & Enhanced Visualization Guide

## Overview

The GRASP data pipeline now includes comprehensive HDF5 file inspection and visualization capabilities. This guide explains how to unpack and understand all information stored in HDF5 episodes.

## What Information is Stored in HDF5 Files?

Each episode HDF5 file contains:

### 1. **Actions** (`/action`)
- Robot control commands at each timestep
- Shape: `(T, action_dim)` where T = episode length
- Data: Continuous joint/gripper commands
- Example: 500 timesteps × 4 action dimensions

### 2. **Robot State - Joint Positions** (`/observations/qpos`)
- Joint angles/positions at each timestep
- Shape: `(T, num_dof)` where num_dof is degrees of freedom
- Data: Full robot configuration state
- Example: 500 timesteps × 16 DOF

### 3. **Robot State - Joint Velocities** (`/observations/qvel`)
- Rate of change of joint angles
- Shape: `(T, num_dof)`
- Data: Joint angular velocities in rad/s
- Example: 500 timesteps × 15 DOF

### 4. **Multi-Camera Images** (`/observations/images/{camera_name}`)
- RGB video frames from selected cameras
- Shape: `(T, H, W, 3)` for each camera
- Data: RGB images (uint8, 0-255)
- Cameras: Selected from configuration (top, left, right, front, gripper)
- Compression: gzip (lossless, ~50-70% reduction)
- Example: 500 frames × 480×640 pixels × 3 channels per camera

### 5. **Task Description** (`/language_raw`)
- Natural language instruction
- Example: "Pick up the red cube and place it in the box"

### 6. **Metadata** (Root Attributes)
- `sim`: Boolean flag (True for simulation, False for real robot)
- File size, compression details, etc.

---

## Using the Enhanced Visualization Tool

The updated `visualize_dataset.py` now unpacks and displays ALL information stored in HDF5 files.

### Basic Usage

#### 1. **Inspect a Single Episode (Information Only)**
```bash
python scripts/visualize_dataset.py --file data/episode_0.hdf5 --info-only
```

Output includes:
- File metadata (size, source, compression)
- Task description
- Action statistics (min, max, mean, std)
- Joint position statistics
- Joint velocity statistics
- Camera information (resolution, compression, storage size)

#### 2. **Visualize Episode with Overlaid Information**
```bash
python scripts/visualize_dataset.py --file data/episode_0.hdf5
```

This shows:
- Multi-camera video feed
- Frame number and task description
- Robot state (qpos values) overlaid on video
- Action commands overlaid on video
- Press 'q' to quit, any other key to speed up playback

#### 3. **Skip Information, Go Straight to Video**
```bash
python scripts/visualize_dataset.py --file data/episode_0.hdf5 --no-info
```

#### 4. **Batch Process Directory**
```bash
python scripts/visualize_dataset.py --dir data/metaworld_dataset/
```

Processes all HDF5 files in the directory sequentially.

#### 5. **Info Only for All Files**
```bash
python scripts/visualize_dataset.py --dir data/metaworld_dataset/ --info-only
```

---

## Example Output

### Information Unpacking Output

```
================================================================================
UNPACKED HDF5 FILE INFORMATION
================================================================================

[METADATA]
  File Path: data/metaworld_pick-place-v3/episode_0.hdf5
  File Size: 35.93 MB
  From Simulation: True
  Compressed: False
  Episode Length: 500 timesteps

[LANGUAGE INSTRUCTION]
  Task: interactions with pick-place-v3

[ACTIONS]
  Shape: (500, 4)
  Data Type: float32
  Dimensions: timesteps=500, action_dim=4
  Statistics:
    Min: -1.0000
    Max: 1.0000
    Mean: -0.1157
    Std: 0.7254

[ROBOT STATE - JOINT POSITIONS (qpos)]
  Shape: (500, 16)
  Data Type: float32
  Dimensions: timesteps=500, dof=16
  Statistics:
    Min: -0.9831
    Max: 2.3906
    Mean: 0.5313
    Std: 0.9118

[ROBOT STATE - JOINT VELOCITIES (qvel)]
  Shape: (500, 15)
  Data Type: float32
  Dimensions: timesteps=500, dof=15
  Statistics:
    Min: -5.8080
    Max: 7.2896
    Mean: 0.0098
    Std: 0.8250

[CAMERA IMAGES]

  Camera: FRONT
    Shape: (500, 480, 640, 3)
    Data Type: uint8
    Dimensions: timesteps=500, height=480, width=640, channels=3
    Uncompressed Size: 439.45 MB [Compression: gzip]

  Camera: TOP
    Shape: (500, 480, 640, 3)
    Data Type: uint8
    Dimensions: timesteps=500, height=480, width=640, channels=3
    Uncompressed Size: 439.45 MB [Compression: gzip]

================================================================================
```

---

## Understanding the Data

### Episode Structure

```
Timesteps:  0      1      2     ...    499
           |------|------|------|------|
Action:    a0     a1     a2     ...    a499
qpos:      q0     q1     q2     ...    q499
qvel:      v0     v1     v2     ...    v499
Images:   [I0]  [I1]   [I2]    ...   [I499]
```

Each timestep contains:
- Control command (action)
- Robot configuration (qpos)
- Robot velocity (qvel)
- Sensor observations (images from all cameras)

### Action Space

For MetaWorld:
- 4D action: x, y, z gripper position deltas + gripper open/close
- Values normalized to [-1, 1]

### Joint State

For 7-DOF Arm + Gripper:
- qpos: 16D (7 joints + gripper state + extra dimensions)
- qvel: 15D (7 joint velocities + gripper velocity + other)

### Images

Each camera captures RGB frames:
- Resolution: 480×640 pixels (configurable)
- Format: uint8 (0-255 range)
- Channels: RGB (3 channels)
- Compression: gzip (lossless)

---

## Advanced Usage: Python API

### Programmatically Unpack an Episode

```python
from visualize_dataset import unpack_hdf5_file

# Extract all information from an HDF5 file
data = unpack_hdf5_file('data/episode_0.hdf5', verbose=True)

# Access individual components
actions = data['action']              # (T, 4) array
qpos = data['observations']['qpos']   # (T, 16) array
qvel = data['observations']['qvel']   # (T, 15) array
images = data['images']['top']        # (T, 480, 640, 3) array
task = data['language']               # "Pick up the red cube..."

# Access statistics
print(f"Action min: {data['action_stats']['min']}")
print(f"Episode length: {data['episode_length']}")
```

### Print HDF5 Structure

```python
from visualize_dataset import print_hdf5_structure

print_hdf5_structure('data/episode_0.hdf5')
```

This displays the complete hierarchical structure with shapes, dtypes, and compression info.

---

## File Size Considerations

### Typical Episode (500 timesteps, 3 cameras)
- Actions: ~8 KB
- Joint states: ~60 KB
- 3 cameras (480×640): ~45-75 MB (with gzip compression)
- **Total: ~45-75 MB per episode**

### Optimization Tips
1. **Reduce image resolution**: Use smaller images (128×128 instead of 480×640)
2. **Select fewer cameras**: Choose only necessary cameras (top + gripper instead of all 5)
3. **Compression is automatic**: Images use gzip by default (no extra steps needed)

---

## Camera Configuration

The cameras stored depend on `camera_config.py`:

```python
from camera_config import SELECTED_CAMERAS, get_camera_names

# Currently selected cameras
cameras = get_camera_names()  # ['front'] by default in your data

# Available cameras
print(SELECTED_CAMERAS)  # Shows which cameras are selected
```

To change which cameras are recorded, edit `camera_config.py`:

```python
SELECTED_CAMERAS = ['top', 'left', 'right', 'front', 'gripper']
```

---

## Visualize Dataset with State Information

The enhanced visualization overlays all extracted information on the video:

```
┌─────────────────────────────────────────────────────────┐
│ Frame: 42/500                                           │
│ Task: Pick up the red cube and place it in the box     │
│ qpos: [ 0.123, 0.456, 0.789, ... ]                     │
│ qvel: [ 0.012, 0.034, -0.001, ... ]                    │
│ action: [ 0.1, 0.2, 0.3, ... ]                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  FRONT (ID:4)    |    TOP (ID:0)                        │
│  [image frame]   |  [image frame]                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Issue: "No images found in file"
- Check which cameras are in the file: `--info-only` shows all stored data
- Camera selection may have changed; file might have different cameras

### Issue: Large file sizes (>100 MB per episode)
- Reduce image resolution during generation
- Select fewer cameras
- Compression is automatic; cannot be reduced further without quality loss

### Issue: Slow visualization
- Use `--no-info` to skip unpacking information
- Reduce number of cameras
- Decrease image resolution

---

## See Also

- [HDF5_STRUCTURE.md](HDF5_STRUCTURE.md) - Complete technical HDF5 format specification
- [camera_config.py](../scripts/camera_config.py) - Configure camera selection
- [generate_metaworld_data.py](../scripts/generate_metaworld_data.py) - Data generation script
- [visualize_dataset.py](../scripts/visualize_dataset.py) - Visualization source code
