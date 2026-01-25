# HuggingFace Training - Quick Reference

## ✅ What Changed

### Files Modified
1. **[src/data/datasets.py](../src/data/datasets.py)** - Added PyArrow compatibility patch
2. **[scripts/camera_config.py](../scripts/camera_config.py)** - Already set to `['right']` for corner2
3. **[scripts/aloha_scripts/constants.py](../scripts/aloha_scripts/constants.py)** - Already has `metaworld_hf` config

### Files Created
1. **[scripts/train_hf_metaworld.sh](../scripts/train_hf_metaworld.sh)** - Training script for HF dataset
2. **[scripts/test_hf_dataset.py](../scripts/test_hf_dataset.py)** - Dataset validation script
3. **[docs/HF_TRAINING_GUIDE.md](HF_TRAINING_GUIDE.md)** - Complete training guide

## 🚀 Quick Start (3 steps)

```bash
# 1. Activate environment
conda activate lerobot

# 2. Test dataset (optional but recommended)
python scripts/test_hf_dataset.py

# 3. Start training
./scripts/train_hf_metaworld.sh
```

## 📊 Dataset Details

| Property | Value |
|----------|-------|
| Dataset | `aadarshram/metaworld-pick-place-v3` |
| Episodes | 50 (100% expert) |
| Frames | 2,661 |
| Camera | corner2 (right view) |
| State Dim | 4 (x, y, z, gripper) |
| Action Dim | 4 (x, y, z, gripper) |
| Image Size | 480x480x3 |

## ⚙️ Key Parameters

```bash
--task_name metaworld_hf           # Use HF dataset config
--action_dim 4                     # Match dataset
--state_dim 4                      # Match dataset
--chunk_size 16                    # Action prediction horizon
--action_head_type droid_diffusion # Diffusion policy
--lora_enable True                 # Use LoRA fine-tuning
```

## 🔍 Verify Setup

```bash
# Check camera config
python scripts/camera_config.py
# Expected: Selected Cameras: ['right']

# Check dataset loading  
python scripts/test_hf_dataset.py
# Expected: All checks passed!

# Check training args
./scripts/train_hf_metaworld.sh --help
```

## 📁 Output Location

```
outputs/metaworld_hf_train/
├── checkpoint-100/
├── adapter_model.safetensors
├── dataset_stats.pkl
└── logs/
```

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| PyArrow error | Already patched in code |
| Dataset download slow | Wait for first time, cached after |
| Memory error | Reduce batch size to 4 |
| Camera mismatch | Already configured correctly |

## 📖 Full Documentation

See [HF_TRAINING_GUIDE.md](HF_TRAINING_GUIDE.md) for complete details.
