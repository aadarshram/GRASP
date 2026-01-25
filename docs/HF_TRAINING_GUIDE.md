# HuggingFace Dataset Training Guide

## Overview
This guide explains how to train the GRASP model using the `aadarshram/metaworld-pick-place-v3` HuggingFace dataset.

## Dataset Information

**Dataset:** `aadarshram/metaworld-pick-place-v3`
- **Total Episodes:** 50 expert trajectories (100% success rate)
- **Total Frames:** 2,661
- **Camera View:** corner2 (right view) only
- **Image Size:** 480x480x3
- **FPS:** 80
- **Task:** Pick and place manipulation

### Data Structure
```python
{
    "observation.image": Image(480, 480, 3),  # RGB image from corner2 camera
    "observation.state": [4],  # [x, y, z, gripper] - end-effector state
    "observation.environment_state": [39],  # Environment keypoints
    "action": [4],  # [x, y, z, gripper] - action commands
    "task_id": [1],  # Task identifier
    "timestamp": [1],  # Frame timestamp
    "frame_index": [1],  # Frame number in episode
    "episode_index": [1],  # Episode number
    "index": [1],  # Global sample index
    "task_index": [1]  # Task index
}
```

## Configuration Changes

### 1. Camera Configuration
The camera is already configured in [`scripts/camera_config.py`](scripts/camera_config.py):
```python
SELECTED_CAMERAS = ['right']  # 'right' corresponds to 'corner2' in MetaWorld
```

### 2. Task Configuration
Added in [`scripts/aloha_scripts/constants.py`](scripts/aloha_scripts/constants.py):
```python
'metaworld_hf': {
    'hf_dataset_name': 'aadarshram/metaworld-pick-place-v3',
    'episode_len': 500,
    'camera_names': get_camera_names(),  # Uses ['right']
    'train_ratio': 0.95,
    'use_hf_dataset': True,
}
```

### 3. Data Loading
The HuggingFace dataset loader is implemented in [`src/data/datasets.py`](src/data/datasets.py):
- `HFEpisodicDataset`: Dataset class for HF data
- `load_hf_data()`: Loads and prepares HF datasets
- `get_norm_stats_hf()`: Computes normalization statistics

## Training

### Quick Start
```bash
# Make sure you're in the lerobot environment
conda activate lerobot

# Run training
./scripts/train_hf_metaworld.sh
```

### Training Script Details
The training script [`scripts/train_hf_metaworld.sh`](scripts/train_hf_metaworld.sh) uses:

**Model Configuration:**
- Base Model: `EleutherAI/pythia-1.4b`
- Action Head: Diffusion policy (`droid_diffusion`)
- LoRA: Enabled (r=64, alpha=256)
- Image Size: 480x480

**Training Configuration:**
- Epochs: 50
- Batch Size: 8
- Learning Rate: 5e-5 (LoRA), 3e-5 (non-LoRA)
- Max Steps: 5,000
- Chunk Size: 16 action steps
- State Dim: 4 (x, y, z, gripper)
- Action Dim: 4 (x, y, z, gripper)

**Hardware:**
- Mixed Precision: BF16
- Gradient Checkpointing: Enabled
- Data Workers: 4

### Custom Training
Modify the training parameters:
```bash
python scripts/train.py \
    --task_name metaworld_hf \
    --output_dir ./outputs/my_experiment \
    --num_train_epochs 50 \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-5 \
    --action_dim 4 \
    --state_dim 4 \
    --chunk_size 16
```

## Implementation Details

### PyArrow Compatibility Fix
The code includes a patch for PyArrow compatibility issues:
```python
# In src/data/datasets.py
import pyarrow as pa
original_concat_tables = pa.concat_tables
def patched_concat_tables(tables, **kwargs):
    kwargs.pop('promote_options', None)
    return original_concat_tables(tables, **kwargs)
pa.concat_tables = patched_concat_tables
```

### Image Processing
Since the HF dataset has only one camera view, the code replicates it for multi-camera compatibility:
```python
num_cams = len(self.camera_names)
all_cam_images = np.stack([image for _ in range(num_cams)], axis=0)
```

### Normalization
- **Actions:** Normalized to [-1, 1] for diffusion policy
- **States:** Normalized to mean=0, std=1
- Statistics computed from the entire training set

## Testing

### Test Dataset Loading
```bash
python scripts/test_hf_dataset.py
```

This will:
1. Load the dataset from HuggingFace
2. Verify data structure
3. Check episode counts
4. Validate dimensions

## Output Structure
```
outputs/metaworld_hf_train/
├── checkpoint-100/          # Saved checkpoints
├── checkpoint-200/
├── ...
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors  # LoRA weights
├── dataset_stats.pkl        # Normalization statistics
├── trainer_state.json       # Training state
└── logs/                    # Training logs
    └── wandb/              # WandB logs
```

## Troubleshooting

### Dataset Download Issues
If the dataset fails to download:
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/aadarshram___metaworld-pick-place-v3
python scripts/test_hf_dataset.py
```

### Memory Issues
Reduce batch size:
```bash
--per_device_train_batch_size 4
```

### PyArrow Errors
The code includes a patch, but if issues persist:
```bash
pip install --upgrade pyarrow datasets
```

## Next Steps

1. **Monitor Training:**
   - Check WandB dashboard for metrics
   - Monitor `outputs/metaworld_hf_train/logs`

2. **Evaluate Model:**
   - Use trained checkpoint for evaluation
   - Test on MetaWorld pick-place task

3. **Fine-tune:**
   - Adjust learning rates
   - Modify chunk size
   - Change LoRA parameters

## References

- Dataset: https://huggingface.co/datasets/aadarshram/metaworld-pick-place-v3
- Generator Script: https://github.com/aadarshram/lerobot/blob/MultiTask/src/lerobot/scripts/generate_MetaWorld_datasets.py
- Base Model: https://huggingface.co/EleutherAI/pythia-1.4b
