#!/bin/bash

# Get absolute paths for all resources
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== GRASP VLA Training ==="


# Configuration
ACTION_HEAD=droid_diffusion  # Options: 'droid_diffusion' or 'act'
echo "Training LLaVA-Pythia with Action Head: $ACTION_HEAD"
echo ""
OUTPUT=outputs/

# Create output directory
if [ -d "$OUTPUT" ]; then
   echo "Output directory exists: $OUTPUT"
else
   echo "Creating output directory: $OUTPUT"
   mkdir -p $OUTPUT
fi

# Backup train script for reproducibility
cp ./scripts/train.sh $OUTPUT
echo "Saved training script to: $OUTPUT/train.sh"
echo ""

# Training with DeepSpeed
# For parameter descriptions, see scripts/train.py
# lesjie/Llava-Pythia-400M is TinyVLA team's pretrained VLM model on Pythia + LLava framework
echo "Starting training with DeepSpeed..."
echo "Using DeepSpeed config: $PROJECT_ROOT/src/llava-pythia/scripts/zero3_offload.json"
deepspeed --master_port 29600 --num_gpus=1 --num_nodes=1 "$SCRIPT_DIR/train.py" \
  --deepspeed "$PROJECT_ROOT/src/llava-pythia/scripts/zero3_offload.json" \
  --lora_enable True \
  --lora_module 'vit llm' \
  --load_pretrain False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --non_lora_lr 2e-5 \
  --task_name "vla_diff_head_lora" \
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
  --fp16 True \
  --output_dir $OUTPUT \
  --max_steps 10 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.005 \
  --lr_scheduler_type "cosine" \
  --logging_steps 10 \
  --tf32 False \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --action_head_type $ACTION_HEAD \
  --use_state True \
  --concat "token_cat" \
  --window_size 6 \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log

echo ""
echo "Training complete!"
echo "Copying preprocessor config to checkpoints..."

# Copy preprocessor config to all checkpoint directories
for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        if [ -f "src/llava-pythia/preprocessor_config.json" ]; then
            cp src/llava-pythia/preprocessor_config.json "$dir"
            echo "Copied to: $dir"
        fi
    fi
done

echo ""
echo "=== Training pipeline finished ==="
echo "Outputs saved to: $OUTPUT"
echo "View logs with: tensorboard --logdir=$OUTPUT/log"