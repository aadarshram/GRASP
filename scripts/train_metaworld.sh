#!/bin/bash

# Get absolute paths for all resources
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"

echo "=== GRASP VLA Training (Metaworld) ==="

# Configuration
ACTION_HEAD=droid_diffusion
echo "Training LLaVA-Pythia on Metaworld Task with Action Head: $ACTION_HEAD"
echo ""
OUTPUT=outputs/metaworld_train

# Create output directory
mkdir -p $OUTPUT

# Backup train script for reproducibility
cp "$0" $OUTPUT/train.sh
echo "Saved training script to: $OUTPUT/train.sh"
echo ""

# Check for latest checkpoint
LATEST_CHECKPOINT=$(ls -td $OUTPUT/checkpoint-* 2>/dev/null | head -1)
if [ -n "$LATEST_CHECKPOINT" ]; then
  echo "Found checkpoint: $LATEST_CHECKPOINT"
  RESUME_ARGS="--resume_from_checkpoint $LATEST_CHECKPOINT"
else
  RESUME_ARGS=""
fi

echo "Starting training with DeepSpeed..."
deepspeed --master_port 29600 --num_gpus=1 --num_nodes=1 "$SCRIPT_DIR/train.py" \
  --deepspeed "$PROJECT_ROOT/src/llava-pythia/scripts/zero3_offload.json" \
  --lora_enable True \
  --lora_module 'llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-6 \
  --task_name "metaworld_task" \
  --model_name_or_path "lesjie/Llava-Pythia-400M" \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 False \
  --fp16 False \
  --tf32 False \
  --output_dir $OUTPUT \
  --max_steps 4000 \
  --per_device_train_batch_size 50 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 400 \
  --save_total_limit 3 \
  --learning_rate 2e-6 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --action_dim 4 \
  --state_dim 16 \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --report_to wandb \
  --run_name "metaworld_overfit_test" \
  --logging_dir $OUTPUT/log \
  $RESUME_ARGS

echo ""
echo "Training complete!"
