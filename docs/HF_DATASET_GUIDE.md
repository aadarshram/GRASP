# HuggingFace Dataset Integration for GRASP

This guide explains how to train GRASP on the HuggingFace dataset `aadarshram/metaworld-pick-place-v3`.

## Dataset Information

- **Dataset**: [aadarshram/metaworld-pick-place-v3](https://huggingface.co/datasets/aadarshram/metaworld-pick-place-v3)
- **Total Episodes**: 50
- **Total Frames**: 2,661
- **Camera View**: corner2 (3rd-person view)
- **Image Resolution**: 480x480x3
- **Action Dimensions**: 4 (x, y, z, gripper)
- **State Dimensions**: 4 (x, y, z, gripper)
- **Task**: Pick and place objects

## Changes Made

### 1. **New Dataset Loader** (`src/data/datasets.py`)
   - Added `HFEpisodicDataset` class to load data from HuggingFace datasets
   - Added `get_norm_stats_hf()` to compute normalization statistics from HF data
   - Added `load_hf_data()` function to handle HF dataset loading and train/val splitting

### 2. **Camera Configuration** (`scripts/camera_config.py`)
   - Updated `SELECTED_CAMERAS` to `['right']` (corresponds to 'corner2' in MetaWorld)
   - This matches the single camera view available in the HF dataset

### 3. **Task Configuration** (`scripts/aloha_scripts/constants.py`)
   - Added `metaworld_hf` task configuration:
     ```python
     'metaworld_hf': {
         'hf_dataset_name': 'aadarshram/metaworld-pick-place-v3',
         'episode_len': 500,
         'camera_names': get_camera_names(),
         'train_ratio': 0.95,
         'use_hf_dataset': True,
     }
     ```

### 4. **Training Script** (`scripts/train.py`)
   - Updated default `action_dim` from 10 to 4
   - Updated default `state_dim` from 7 to 4
   - Modified `main()` to detect and use HuggingFace datasets when configured
   - Added conditional loading: uses `load_hf_data()` for HF datasets, `load_data()` for local HDF5

### 5. **Training Shell Script** (`scripts/train_hf.sh`)
   - Created new training script specifically for the HuggingFace dataset
   - Configured with appropriate hyperparameters:
     - Image size: 480x480
     - Batch size: 4
     - Action/State dimensions: 4
     - Max steps: 5000
     - Learning rate: 2e-4

## Requirements

Make sure you have the HuggingFace `datasets` library installed:

```bash
pip install datasets
```

This should already be in your environment, but if not:

```bash
conda activate grasp  # or your environment name
pip install datasets
```

## Usage

### Quick Start

Run the training script:

```bash
cd /home/hemanthm/Desktop/GRASP/GRASP
bash scripts/train_hf.sh
```

### Configuration Options

The training script uses the following key parameters:

- **Task Name**: `metaworld_hf` (defined in constants.py)
- **Model**: `lesjie/Llava-Pythia-400M` (pre-trained VLM)
- **Action Head**: `droid_diffusion` (can be changed to `act`)
- **LoRA**: Enabled for efficient fine-tuning
- **Image Size**: 480x480 (matches HF dataset)
- **Batch Size**: 4 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Training Steps**: 5000

### Customization

To modify training parameters, edit `scripts/train_hf.sh`:

```bash
# Change batch size
--per_device_train_batch_size 8 \

# Change learning rate
--learning_rate 1e-4 \

# Change max steps
--max_steps 10000 \

# Change action head type
ACTION_HEAD=act  # or droid_diffusion
```

### Switching Between Datasets

To switch back to local HDF5 datasets:

1. Edit `scripts/camera_config.py` to select desired cameras
2. Use the original `train.sh` script or modify `task_name` parameter
3. Run: `bash scripts/train.sh`

## Data Flow

```
HuggingFace Dataset (aadarshram/metaworld-pick-place-v3)
    ↓
load_hf_data() - Download and prepare dataset
    ↓
get_norm_stats_hf() - Compute normalization statistics
    ↓
HFEpisodicDataset - PyTorch Dataset wrapper
    ↓
Train/Val Split (95%/5%)
    ↓
DataLoader → Model Training
```

## Model Architecture

The system uses the same architecture as the original GRASP:

```
Multi-Camera Images → Vision Tower (CLIP/SigLIP) → MM Projector (MLP)
                                                           ↓
Language Instruction → Tokenizer → Pythia LLM ← ← ← ← ← ←
                                      ↓
Robot State (qpos) → → → → → → Action Head (Diffusion) → Action Chunk (16 steps)
```

**Key Differences for HF Dataset**:
- Single camera view (corner2) instead of multiple cameras
- 4D actions (x, y, z, gripper) instead of 10D
- 4D state (x, y, z, gripper) instead of 7D

## Output

Training outputs will be saved to:
- **Checkpoints**: `outputs/metaworld_hf_train/checkpoint-*`
- **Logs**: `outputs/metaworld_hf_train/log/`
- **Dataset Stats**: `outputs/metaworld_hf_train/dataset_stats.pkl`

View training logs:
```bash
tensorboard --logdir=outputs/metaworld_hf_train/log
```

## Troubleshooting

### Issue: `datasets` module not found
```bash
pip install datasets
```

### Issue: CUDA out of memory
Reduce batch size in `train_hf.sh`:
```bash
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
```

### Issue: Dataset download fails
Check internet connection and HuggingFace access:
```bash
huggingface-cli login
```

### Issue: Camera mismatch
Ensure `camera_config.py` has `SELECTED_CAMERAS = ['right']`

## Validation

After implementing the changes, the code should:
1. ✅ Automatically detect HF dataset when `use_hf_dataset=True`
2. ✅ Download and cache the dataset from HuggingFace
3. ✅ Compute normalization statistics from the HF data
4. ✅ Split data into 95% train, 5% validation
5. ✅ Handle 4D actions and states correctly
6. ✅ Process single camera view (corner2)
7. ✅ Train the model with specified hyperparameters

## Next Steps

1. **Run Training**: `bash scripts/train_hf.sh`
2. **Monitor Progress**: Use TensorBoard or W&B
3. **Evaluate Model**: Use evaluation scripts with trained checkpoints
4. **Fine-tune**: Adjust hyperparameters based on initial results

## Additional Notes

- The HF dataset has 100% expert trajectories
- All episodes are successful demonstrations
- The dataset uses corner2 camera only
- Frame rate: 80 FPS
- Total dataset size: ~300MB (data + videos)
