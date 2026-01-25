#!/bin/bash
# Training script for HuggingFace metaworld-pick-place-v3 dataset
# Dataset: aadarshram/metaworld-pick-place-v3
# Camera: corner2 (right view)
# Episodes: 50 expert trajectories

set -e

# Configuration
MODEL_PATH="EleutherAI/pythia-1.4b"  # Base model
OUTPUT_DIR="./outputs/metaworld_hf_train"
TASK_NAME="metaworld_hf"
NUM_EPOCHS=50
BATCH_SIZE=8
EVAL_BATCH_SIZE=8
LEARNING_RATE=5e-5
NON_LORA_LR=3e-5
MAX_STEPS=5000
LOGGING_STEPS=10
SAVE_STEPS=100
EVAL_STEPS=200

# Action and state dimensions for metaworld HF dataset
ACTION_DIM=4  # [x, y, z, gripper]
STATE_DIM=4   # [x, y, z, gripper]
CHUNK_SIZE=16
WINDOW_SIZE=6

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run training
python scripts/train.py \
    --model_name_or_path "$MODEL_PATH" \
    --task_name "$TASK_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --non_lora_lr "$NON_LORA_LR" \
    --max_steps "$MAX_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --action_dim "$ACTION_DIM" \
    --state_dim "$STATE_DIM" \
    --chunk_size "$CHUNK_SIZE" \
    --window_size "$WINDOW_SIZE" \
    --action_head_type "droid_diffusion" \
    --use_state True \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --lora_module "vit llm" \
    --freeze_vision_tower False \
    --freeze_backbone False \
    --pretrain_image_size 480 \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --do_eval False \
    --seed 42 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to "wandb" \
    --run_name "metaworld_hf_$(date +%Y%m%d_%H%M%S)"

echo "Training completed! Model saved to: $OUTPUT_DIR"
