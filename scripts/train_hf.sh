#!/bin/bash
# filepath: /home/hemanthm/Desktop/GRASP/GRASP/scripts/train_hf.sh
# GRASP VLA Training for HuggingFace metaworld-pick-place-v3 Dataset
# Single unified training script with all configurations

set -e

# Get paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure we're in the lerobot conda environment
if [[ "$CONDA_DEFAULT_ENV" != "lerobot" ]]; then
    echo "❌ Error: lerobot conda environment not activated!"
    echo "Please run: conda activate lerobot"
    exit 1
fi

# Use conda environment's deepspeed
DEEPSPEED_CMD="$CONDA_PREFIX/bin/deepspeed"

echo "=========================================="
echo "GRASP VLA Training - MetaWorld HF Dataset"
echo "=========================================="
echo ""

# ========================
# CONFIGURATION
# ========================
MODEL_NAME="lesjie/Llava-Pythia-400M"
OUTPUT_DIR="./outputs/metaworld_hf_train"
TASK_NAME="metaworld_hf"
ACTION_HEAD="droid_diffusion"

# Training hyperparameters
NUM_EPOCHS=50
MAX_STEPS=5000
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
LEARNING_RATE=2e-4
NON_LORA_LR=2e-5
WARMUP_RATIO=0.03

# LoRA configuration
LORA_R=64
LORA_ALPHA=256
LORA_MODULE="vit llm"

# Model dimensions for metaworld dataset
ACTION_DIM=4
STATE_DIM=4
CHUNK_SIZE=16
WINDOW_SIZE=6
IMAGE_SIZE=480

# ========================
# SETUP
# ========================
mkdir -p "$OUTPUT_DIR"

# Backup training script for reproducibility
cp "$SCRIPT_DIR/train_hf.sh" "$OUTPUT_DIR/train_hf.sh"
echo "✓ Training configuration saved"
echo ""

# ========================
# TRAINING
# ========================
echo "Starting training with DeepSpeed..."
echo "Dataset: aadarshram/metaworld-pick-place-v3"
echo "Camera: corner2 (right view)"
echo "Episodes: 50 | Frames: 2661"
echo ""

$DEEPSPEED_CMD --master_port 29600 --num_gpus=1 --num_nodes=1 \
  "$SCRIPT_DIR/train.py" \
  --deepspeed "$PROJECT_ROOT/src/llava-pythia/scripts/zero3_offload.json" \
  \
  --model_name_or_path "$MODEL_NAME" \
  --task_name "$TASK_NAME" \
  --output_dir "$OUTPUT_DIR" \
  \
  --lora_enable True \
  --lora_module "$LORA_MODULE" \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --non_lora_lr $NON_LORA_LR \
  \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --pretrain_image_size $IMAGE_SIZE \
  \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  \
  --action_head_type "$ACTION_HEAD" \
  --action_dim $ACTION_DIM \
  --state_dim $STATE_DIM \
  --chunk_size $CHUNK_SIZE \
  --window_size $WINDOW_SIZE \
  --use_state True \
  --concat "token_cat" \
  \
  --num_train_epochs $NUM_EPOCHS \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  \
  --learning_rate $LEARNING_RATE \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type "cosine" \
  --weight_decay 0.0 \
  \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 10 \
  --logging_steps 10 \
  \
  --bf16 True \
  --tf32 True \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --remove_unused_columns False \
  \
  --model_max_length 2048 \
  --logging_dir "$OUTPUT_DIR/log" \
  --report_to wandb \
  --run_name "metaworld-hf-$(date +%Y%m%d_%H%M%S)" \
  --seed 42

# ========================
# POST-TRAINING
# ========================
echo ""
echo "Training complete!"
echo "Copying preprocessor config to checkpoints..."

for dir in "$OUTPUT_DIR"/*/; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        if [ -f "src/llava-pythia/preprocessor_config.json" ]; then
            cp src/llava-pythia/preprocessor_config.json "$dir"
            echo "  ✓ $(basename "$dir")"
        fi
    fi
done

# ========================
# SUMMARY
# ========================
echo ""
echo "=========================================="
echo "✓ Training Pipeline Complete"
echo "=========================================="
echo ""
echo "📊 Results saved to: $OUTPUT_DIR"
echo "📈 Monitor training:"
echo "   tensorboard --logdir=$OUTPUT_DIR/log"
echo ""
echo "📋 Dataset Info:"
echo "   • Name: aadarshram/metaworld-pick-place-v3"
echo "   • Episodes: 50 (all expert, 100% success)"
echo "   • Frames: 2,661"
echo "   • Camera: corner2 (right)"
echo "   • Action: [x, y, z, gripper]"
echo "   • State: [x, y, z, gripper]"
echo ""