# Generalized Robotic Action models for Sensorimotor Policies (GRASP)

[Ongoing project]

Exploring robotic foundation models for general-purpose pick&place manipulation using lerobot and so101 as a part of student robotics club, iBot at Indian Institute of Technology Madras.

## Goal
Most VLAs are data hungry even for simplest of learning and adaptation. Desire to explore data-efficient robot learning for general purpose pick and place manipulation. Will test on lerobot and so101.

## TODO
- Implement a simple VLA first by reproducing any prior work.

## Steps to run

### Step 1: Configure Your Data

Edit `scripts/aloha_scripts/constants.py`:

```python
TASK_CONFIGS = {
    'example_task_config': {
        'dataset_dir': [
            "/YOUR/PATH/TO/hdf5_files",  # ← Change this!
        ],
        'episode_len': 1000,
        'camera_names': ['left', 'right', 'top'],  # ← Match your HDF5 keys!
    },
}
```

**Your HDF5 files should have:**
```python
/observations/qpos          # Robot joint positions
/observations/images/left   # Camera images (match camera_names)
/observations/images/right
/observations/images/top
/action                     # Action sequences
/language_raw              # Task descriptions
```

## Install
dependencies for conda and pip - environment.yml and requirements.txt

pip install -e. - for the main package and llava-pythia



## 📊 Dummy Dataset 


### Dataset Statistics
- **Episodes**: 20 files (19 train, 1 validation)
- **Timesteps per episode**: 100
- **Total timesteps**: 2,000
- **Dataset size**: ~5.3 GB
- **Cameras**: 3 views (left, right, top)
- **Action dim**: 10
- **State dim**: 7
- **Image size**: 480x640 → resized to 180x320

### HDF5 File Structure
```
episode_XXXX.hdf5
├── /action (100, 10)                    # Robot actions
├── /observations/
│   ├── qpos (100, 7)                   # Joint positions
│   ├── qvel (100, 7)                   # Joint velocities  
│   └── images/
│       ├── left (100, 480, 640, 3)    # Left camera RGB
│       ├── right (100, 480, 640, 3)   # Right camera RGB
│       └── top (100, 480, 640, 3)     # Top camera RGB
├── /language_raw                       # Task description string
└── Attributes:
    ├── sim: True                       # From simulation
    └── compress: False                 # Uncompressed images
```

## 🔧 Scripts Available

### 1. Generate Dummy Data
```bash
python scripts/generate_dummy_data.py \
  --output_dir data/my_task \
  --num_episodes 50 \
  --episode_length 200 \
  --action_dim 10 \
  --state_dim 7 \
  --validate
```

**Options:**
- `--compress`: Enable JPEG compression (not working yet, use uncompressed)
- `--image_height/width`: Set image dimensions
- `--validate`: Validate first episode after generation

### 2. Validate Dataset
```bash
python scripts/validate_dataset.py --data_dir data/my_task --verbose
```

Checks:
- Required HDF5 structure
- Shape consistency
- Language instructions
- Camera images
- Attributes

## 📝 Collecting Real Robot Data

### Requirements for Your Own Dataset

#### A. Robot State Data
- **qpos**: Joint positions at each timestep
  - Shape: (T, state_dim)
  - Example: [j1_angle, j2_angle, ..., gripper_pos]
  
- **qvel**: Joint velocities at each timestep
  - Shape: (T, state_dim)
  - Can compute from qpos if not available

#### B. Action Data
- **action**: Target actions or commands
  - Shape: (T, action_dim)
  - Position control: target joint positions
  - Velocity control: velocity commands
  - Include gripper commands

#### C. Multi-Camera Images
- **3 synchronized cameras** (left, right, top)
- RGB images at 30-50 Hz
- Any resolution (will be resized)
- Synchronized with robot state

#### D. Language Instructions
- Natural language task description
- One per episode
- Example: "pick up the red block and place it in the box"


### Run Training

```bash
# Make script executable
chmod +x scripts/train.sh

# Start training
bash scripts/train.sh
```

## 📊 Monitor Training

```bash
# View TensorBoard logs
tensorboard --logdir=outputs/log

# Check training progress
tail -f outputs/log/events.*
```