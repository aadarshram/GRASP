# Complete HDF5 Information Extraction Summary

## What Information is Stored in HDF5 Files?

This document provides a complete reference for all information that can be extracted from GRASP HDF5 episode files.

---

## 1. ACTIONS

### Location in HDF5
`/action`

### Structure
```
Shape: (T, action_dim)
Data Type: float32
Compression: None (raw)
```

### What It Contains
- **T**: Episode length (number of control steps)
- **action_dim**: Dimensionality of action space (typically 4 for MetaWorld)

### MetaWorld Action Details
| Index | Description | Range | Units |
|-------|-------------|-------|-------|
| 0 | Gripper X position delta | [-1, 1] | Normalized |
| 1 | Gripper Y position delta | [-1, 1] | Normalized |
| 2 | Gripper Z position delta | [-1, 1] | Normalized |
| 3 | Gripper open/close | [-1, 1] | Normalized |

### How It's Generated
Actions are control commands sent to the robot at each timestep during the episode.

### Example Usage
```python
import h5py
with h5py.File('episode_0.hdf5', 'r') as f:
    actions = f['/action'][:]  # Shape: (500, 4)
    action_at_t = f['/action'][100]  # Action at timestep 100
```

### Statistics Available
- Minimum action value across entire episode
- Maximum action value across entire episode
- Mean action value
- Standard deviation

---

## 2. ROBOT STATE - JOINT POSITIONS (qpos)

### Location in HDF5
`/observations/qpos`

### Structure
```
Shape: (T, state_dim)
Data Type: float32
Compression: None (raw)
```

### What It Contains
- **T**: Episode length
- **state_dim**: Degrees of freedom (typically 16 for MetaWorld with gripper)

### MetaWorld qpos Details
| Index | Description | Range | Units |
|-------|-------------|-------|-------|
| 0-6 | Joint angles | Variable | Radians |
| 7-9 | Gripper position | Variable | Meters |
| 10+ | Other state variables | Variable | Various |

### Interpretation
- **Joint angles (0-6)**: Position of each robot joint
- **Gripper position (7-9)**: XYZ coordinates of gripper in 3D space
- **Additional state**: Table position, object position, etc.

### Example Usage
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    qpos = f['/observations/qpos'][:]  # Shape: (500, 16)
    joint_angles = qpos[:, 0:7]        # Just the joint angles
    gripper_xyz = qpos[:, 7:10]        # Just gripper position
```

### Statistics Available
- Minimum joint position across episode
- Maximum joint position
- Mean position
- Standard deviation (variation)

---

## 3. ROBOT STATE - JOINT VELOCITIES (qvel)

### Location in HDF5
`/observations/qvel`

### Structure
```
Shape: (T, vel_dim)
Data Type: float32
Compression: None (raw)
```

### What It Contains
- **T**: Episode length
- **vel_dim**: Number of velocity dimensions (typically 15)

### MetaWorld qvel Details
| Index | Description | Range | Units |
|-------|-------------|-------|-------|
| 0-6 | Joint angular velocities | Variable | rad/s |
| 7-14 | Other velocity components | Variable | Various |

### Interpretation
- Shows how fast joints are moving at each timestep
- Zero velocity = stationary
- Positive = moving in positive direction
- Negative = moving in negative direction

### Example Usage
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    qvel = f['/observations/qvel'][:]        # Shape: (500, 15)
    joint_velocities = qvel[:, 0:7]          # rad/s for each joint
    max_joint_speed = np.max(np.abs(qvel[:, 0:7]))
```

### Relationship to qpos
- `qvel` is the time derivative of `qpos`
- Used for dynamics models and physics simulation
- Important for understanding motion patterns

---

## 4. CAMERA IMAGES

### Location in HDF5
`/observations/images/{camera_name}`

### Structure
```
Shape: (T, H, W, 3)
Data Type: uint8
Compression: gzip (lossless)
```

### What It Contains
- **T**: Episode length (number of frames)
- **H**: Image height (typically 480 pixels)
- **W**: Image width (typically 640 pixels)
- **3**: RGB channels (Red, Green, Blue)

### Available Cameras

| Camera Name | Camera ID | View | Use Case |
|------------|-----------|------|----------|
| `top` | 0 | Bird's eye view | Global context, object locations |
| `left` | 1 | Left side view | Manipulation visibility |
| `right` | 2 | Right side view | Alternative angle, collision detection |
| `front` | 4 | Front view | Hand-object interaction detail |
| `gripper` | 5 | Egocentric (gripper mounted) | Fine-grained grasping details |

### Image Format
- **Color Space**: RGB (0-255 for each channel)
- **Compression**: gzip lossless (~50-70% size reduction)
- **Quality**: No loss - full RGB information preserved

### Example Usage
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    top_images = f['/observations/images/top'][:]           # (500, 480, 640, 3)
    single_frame = f['/observations/images/top'][100]       # Frame at t=100
    single_pixel = f['/observations/images/top'][100, 240, 320, 0]  # Red channel
```

### Visualization
```python
import cv2

with h5py.File('episode_0.hdf5', 'r') as f:
    img = f['/observations/images/top'][50]  # Get frame 50
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Frame 50', img_bgr)
    cv2.waitKey(0)
```

### Storage Considerations
- Single camera, single episode: ~220 MB uncompressed
- With gzip: ~110 MB
- Multiple cameras multiply the size accordingly

---

## 5. LANGUAGE INSTRUCTION

### Location in HDF5
`/language_raw`

### Structure
```
Shape: (1,)
Data Type: UTF-8 variable-length string
Compression: None
```

### What It Contains
Natural language description of the task in the episode.

### Examples
```
"Pick up the red cube and place it on the green square"
"Push the block to the left"
"Open the drawer completely"
"Stack all three blocks in the correct order"
```

### Format
- Single string value
- UTF-8 encoded
- Human-readable task description
- Typically 5-100 characters

### Example Usage
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    task_bytes = f['/language_raw'][0]
    task_str = task_bytes.decode('utf-8') if isinstance(task_bytes, bytes) else str(task_bytes)
    print(f"Task: {task_str}")
```

### Relationship to Action/State
- Describes the objective of the entire episode
- All actions and state changes are in service of completing this task
- One language instruction per episode

---

## 6. METADATA & ATTRIBUTES

### Root Attributes Location
Stored at the root level of the HDF5 file (not in any group)

### Available Attributes

| Attribute | Type | Example | Meaning |
|-----------|------|---------|---------|
| `sim` | Boolean | `True` | Whether data is from simulation (True) or real robot (False) |

### Example Usage
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    is_simulation = f.attrs['sim']
    print(f"Simulation data: {is_simulation}")
```

### File-Level Information
```python
import os
with h5py.File('episode_0.hdf5', 'r') as f:
    file_size_mb = os.path.getsize('episode_0.hdf5') / (1024**2)
    num_timesteps = f['/action'].shape[0]
    action_dim = f['/action'].shape[1]
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Episode length: {num_timesteps} timesteps")
```

---

## Complete Information Extraction Checklist

Use this checklist to verify you're accessing all available HDF5 data:

- [ ] **Episode Length**: `f['/action'].shape[0]` → Number of timesteps
- [ ] **Action Dimension**: `f['/action'].shape[1]` → Action space dimensionality
- [ ] **Action Data**: `f['/action'][:]` → Complete action sequence
- [ ] **Joint Positions**: `f['/observations/qpos'][:]` → Robot configuration
- [ ] **Joint Velocities**: `f['/observations/qvel'][:]` → Robot motion state
- [ ] **Available Cameras**: `list(f['/observations/images'].keys())` → Which cameras are stored
- [ ] **Image Resolution**: `f['/observations/images/top'].shape` → Image dimensions
- [ ] **Task Description**: `f['/language_raw'][0]` → Instruction/goal
- [ ] **Data Source**: `f.attrs['sim']` → Simulation vs. real robot
- [ ] **Compression Info**: `f['/observations/images/top'].compression` → How data is compressed

---

## Summary Statistics You Can Compute

### From Actions
```python
with h5py.File('episode_0.hdf5', 'r') as f:
    action = f['/action'][:]
    stats = {
        'min': np.min(action),
        'max': np.max(action),
        'mean': np.mean(action),
        'std': np.std(action),
        'range': np.max(action) - np.min(action),
    }
```

### From Joint States
```python
    qpos = f['/observations/qpos'][:]
    # Per-DOF statistics
    for dof in range(qpos.shape[1]):
        dof_trajectory = qpos[:, dof]
        print(f"DOF {dof}: min={np.min(dof_trajectory):.3f}, max={np.max(dof_trajectory):.3f}")
```

### From Images
```python
    images = f['/observations/images/top'][:]
    # Image statistics
    print(f"Mean brightness: {np.mean(images):.1f}")
    print(f"Image variation: {np.std(images):.1f}")
```

---

## Data Flow in Episode

```
Time progression (0 → T-1):
├─ Timestep 0
│  ├─ action[0] → control command sent to robot
│  ├─ qpos[0] → robot position after executing action
│  ├─ qvel[0] → robot velocity
│  └─ images/*/[0] → visual observations from all cameras
│
├─ Timestep 1
│  ├─ action[1] → next control command
│  ├─ qpos[1] → updated robot position
│  ├─ qvel[1] → updated velocity
│  └─ images/*/[1] → updated images
│
└─ Timestep T-1
   ├─ action[T-1]
   ├─ qpos[T-1]
   ├─ qvel[T-1]
   └─ images/*/[T-1]
```

Each row/frame is synchronized - they all correspond to the same timestep.

---

## Using visualize_dataset.py to Extract Information

### Print Full Structure
```bash
python scripts/visualize_dataset.py --file data/episode_0.hdf5 --info-only
```

### View with Visualization
```bash
python scripts/visualize_dataset.py --file data/episode_0.hdf5
```

This will:
1. Print all information statistics
2. Show task description
3. Display video with overlaid state/action data
4. Allow frame-by-frame inspection

---

## Data Types and Precision

| Data | Type | Precision | Range |
|------|------|-----------|-------|
| Actions | float32 | ~7 decimal places | [-1, 1] typically |
| qpos | float32 | ~7 decimal places | Varies by joint |
| qvel | float32 | ~7 decimal places | Varies by joint |
| Images | uint8 | Integer | [0, 255] |
| Language | String | N/A | Text |
| sim flag | Boolean | N/A | True/False |

---

## File Size Breakdown (Example)

For a 500-timestep episode with 3 cameras (480×640):

| Component | Size (Uncompressed) | Size (Compressed) | Notes |
|-----------|-------------------|------------------|-------|
| actions | 8 KB | 8 KB | Not compressed |
| qpos | 32 KB | 32 KB | Not compressed |
| qvel | 30 KB | 30 KB | Not compressed |
| 3 cameras | ~1314 MB | ~39 MB | gzip ~50-70% reduction |
| language | <1 KB | <1 KB | Tiny |
| **Total** | **~1314 MB** | **~39-75 MB** | Depends on scene |

---

## Complete Usage Example

```python
import h5py
import numpy as np
from camera_config import get_camera_names

# Open file and extract everything
with h5py.File('data/episode_0.hdf5', 'r') as f:
    # Metadata
    is_sim = f.attrs['sim']
    task = f['/language_raw'][0].decode('utf-8')
    T = f['/action'].shape[0]
    
    # Extract all sequences
    actions = f['/action'][:]              # (T, 4)
    qpos = f['/observations/qpos'][:]      # (T, 16)
    qvel = f['/observations/qvel'][:]      # (T, 15)
    
    # Extract images from selected cameras
    cameras = get_camera_names()
    images = {}
    for cam in cameras:
        if f'/observations/images/{cam}' in f:
            images[cam] = f[f'/observations/images/{cam}'][:]  # (T, H, W, 3)
    
    # Print summary
    print(f"Task: {task}")
    print(f"Episode length: {T} timesteps")
    print(f"Action space: {actions.shape[1]}D")
    print(f"State space (positions): {qpos.shape[1]}D")
    print(f"Cameras: {list(images.keys())}")
    print(f"Image resolution: {images[cameras[0]].shape[1:3]}")
    
    # Compute statistics
    print(f"\nAction statistics:")
    print(f"  Min: {np.min(actions):.3f}")
    print(f"  Max: {np.max(actions):.3f}")
    print(f"  Mean: {np.mean(actions):.3f}")
    
    # Visualize a frame
    frame_idx = 100
    for cam in cameras:
        img = images[cam][frame_idx]  # Shape: (480, 640, 3)
        print(f"{cam.capitalize()} camera image at frame {frame_idx}: {img.shape}")
```

---

## See Also
- [HDF5_STRUCTURE.md](HDF5_STRUCTURE.md) - Technical specification
- [HDF5_UNPACKING_GUIDE.md](HDF5_UNPACKING_GUIDE.md) - Usage guide
- [visualize_dataset.py](../scripts/visualize_dataset.py) - Source code for unpacking
