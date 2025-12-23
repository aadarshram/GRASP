import sys
import os
import argparse
import torch
import numpy as np
os.environ['MUJOCO_GL'] = 'egl'
import cv2
import metaworld
import metaworld.policies
from collections import deque
from PIL import Image

# Add paths
script_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, repo_root)
sys.path.insert(0, script_dir)
sys.path.insert(0, os.path.join(repo_root, 'src/llava-pythia'))
sys.path.insert(0, os.path.join(repo_root, 'src'))

import transformers
from transformers import CLIPImageProcessor
from llava_pythia.llava_pythia_utils import load_llava_pythia
from llava_pythia.model.language_model.pythia.llava_pythia import LlavaPythiaConfig
from llava_pythia.constants import DEFAULT_IMAGE_TOKEN
from llava_pythia.mm_utils import tokenizer_image_token

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained checkpoint (output_dir)")
    parser.add_argument("--base_model", type=str, default="lesjie/Llava-Pythia-400M")
    parser.add_argument("--env_name", type=str, default="pick-place-v3")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=4)
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Load Config & Model
    print(f"Loading model from {args.ckpt_path}...")
    
    # Mock config dictionaries expected by load_llava_pythia
    # We need to reconstruct the training config structure
    class DictObj:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    model_args = DictObj(
        model_name_or_path=args.base_model,
        version="v0",
        tune_mm_mlp_adapter=True,
        freeze_vision_tower=True,
        freeze_backbone=True,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        concat="token_cat",
        model_pretrain=""
    )
    training_args = DictObj(
        load_pretrain=False,
        lora_enable=True, # We are loading a LoRA adapter
        lora_r=64,
        lora_alpha=256,
        lora_dropout=0.05,
        lora_bias="none",
        lora_module="llm",
        lora_task_type="CAUSAL_LM",
        bits=16, # Assume bf16/fp16
        bf16=True,
        fp16=False,
        device="cuda",
        gradient_checkpointing=False,
        non_lora_lr=2e-5,
        cache_dir=None,
        tune_mm_mlp_adapter=True,
        freeze_backbone=True,
        freeze_vision_tower=True
    )
    data_args = DictObj(
        is_multimodal=True,
        image_aspect_ratio="pad",
        mm_use_im_start_end=False,
        image_processor=None # Will be loaded
    )
    
    # We cheat slightly: load_llava_pythia usually loads base + applies LoRA.
    # But since we have a checkpoint with 'adapter_model.bin' (LoRA) and potentially non-lora weights,
    # we need to ensure we load the base model first, then the checkpoint.
    
    # Actually, for evaluation, we can use the same loading logic as train.py but 
    # instead of initializing fresh LoRA, we load the saved one.
    
    # However, 'load_llava_pythia' in 'llava_pythia_utils.py' is designed for training setup.
    # It initializes a fresh LoRA config if 'load_pretrain' is False.
    # If we want to load a TRAINED checkpoint, we should treat it as 'load_pretrain=True' 
    # OR let it init fresh and then overwrite with PeftModel.from_pretrained.
    
    # Let's try the standard path: Load base model, then load adapter.
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        model_max_length=2048
    )
    tokenizer.pad_token_id = 1 # As per training log

    llava_pythia_config = LlavaPythiaConfig.from_pretrained(args.base_model, trust_remote_code=True)
    llava_pythia_config.action_head_type = 'droid_diffusion'
    llava_pythia_config.action_dim = args.action_dim
    llava_pythia_config.state_dim = 7
    llava_pythia_config.chunk_size = 16 # Default in train.py
    llava_pythia_config.concat = "token_cat"

    # Load Base Model + LoRA init (randomly initialized LoRA layers)
    config = {
        'model_args': model_args,
        'training_args': training_args,
        'data_args': data_args,
        'bnb_model_from_pretrained_args': {}
    }
    
    model, _ = load_llava_pythia(config=config, llava_pythia_config=llava_pythia_config, tokenizer=tokenizer)
    
    # Now load the trained LoRA weights from ckpt_path
    from peft import PeftModel
    print(f"Loading LoRA weights from {args.ckpt_path}...")
    # The model returned by load_llava_pythia is already a PeftModel (if lora_enable=True)
    # We need to load the state dict from the checkpoint and load it into the model.
    # OR: simpler approach for inference: Use PeftModel.from_pretrained directly on the base model.
    # But load_llava_pythia handles a lot of complexity (vision tower, etc).
    
    # The 'load_llava_pythia' returns a model where 'model.get_model()' is the base.
    # But wait, if training_args.lora_enable is True, it returns a PeftModel wrapping the base.
    # We can try to load_adapter.
    
    # Actually, let's look at how we saved it. 
    # train.py saves with: model.save_pretrained(input_dir) -> saves adapter_model.bin and config.json
    # AND it saves non_lora_trainables.bin (projector, head).
    
    # So we need to:
    # 1. Load non-LoRA trainables (projector, action head).
    # 2. Load the LoRA adapter.
    
    # Load non-LoRA weights
    non_lora_path = os.path.join(args.ckpt_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        # Check parent directory (common output_dir pattern)
        parent_dir = os.path.dirname(args.ckpt_path.rstrip('/'))
        non_lora_path = os.path.join(parent_dir, "non_lora_trainables.bin")

    if os.path.exists(non_lora_path):
        print(f"Loading non-LoRA weights from {non_lora_path}...")
        non_lora_state_dict = torch.load(non_lora_path)
        # We need to handle keys potentially prefixed with 'base_model.model.' if they were saved wrapped
        # But usually 'non_lora_trainables' are saved directly.
        
        # Check keys
        new_state_dict = {}
        for k, v in non_lora_state_dict.items():
            # Remove 'base_model.model.' if present (standard PEFT artifact)
            k = k.replace('base_model.model.', '')
            new_state_dict[k] = v
            
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("Warning: non_lora_trainables.bin not found! Model might be untrained.")

    # Load LoRA weights
    # Since model is already a PeftModel (initialized in load_llava_pythia), we can load parameters.
    # Or cleaner: load params.
    model.load_adapter(args.ckpt_path, adapter_name="default")
    
    model.cuda()
    model.eval()
    
    # Image Processor
    image_processor = CLIPImageProcessor.from_pretrained(args.base_model)
    crop_size = image_processor.crop_size
    
    # 2. Setup Environment
    print(f"Initializing Metaworld env: {args.env_name}")
    ml1 = metaworld.ML1(args.env_name)
    env = ml1.train_classes[args.env_name](render_mode='rgb_array')
    env.set_task(ml1.train_tasks[0])
    
    successes = 0
    frames = []
    
    # 3. Eval Loop
    for ep in range(args.num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
        
        episode_frames = []
        
        # Initial robot state
        # In this specific simplified setup, we might rely on the environment's observation (which contains qpos)
        # OR we need to access output of 'env.sim.data.qpos'.
        # For Metaworld, obs has 39 dims. First 4 are hand pos?
        # Let's trust 'env.sim.data.qpos' for proprioception as used in generator.
        
        step = 0
        done = False
        
        while not done and step < 500:
            # Capture Images
            img = env.render()
            img = cv2.resize(img, (320, 180)) # Match training aspect ratio/size? 
            
            # Save frame for video
            # Convert RGB to BGR for OpenCV
            episode_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            img_pil = Image.fromarray(img)
            # Preprocess image
            image_tensor = image_processor.preprocess(img_pil, return_tensors='pt')['pixel_values'][0]
            
            # Stack images (left, right, top) - we just duplicate for now as per generator
            images = image_tensor.unsqueeze(0).cuda().to(torch.bfloat16)
            images_r = image_tensor.unsqueeze(0).cuda().to(torch.bfloat16)
            images_top = image_tensor.unsqueeze(0).cuda().to(torch.bfloat16)
            
            # Text Prompt
            prompt = f"interactions with {args.env_name}\n"
            # Add image tokens
            # If start_end is False: <image>
            context = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            
            input_ids = tokenizer_image_token(context, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            
            # Proprioception
            # qpos is 7-dim
            try:
                qpos = env.sim.data.qpos.flat[:].copy()[:7] # first 7 are robot
            except:
                qpos = np.zeros(7)
            state_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0).to(torch.bfloat16)
            
            # Inference
            with torch.no_grad():
                # forward(..., eval=True) returns action
                action_chunk = model(
                    input_ids=input_ids,
                    images=images,
                    images_r=images_r,
                    images_top=images_top,
                    states=state_tensor,
                    eval=True
                )
            
            # action_chunk shape: [1, chunk_size, action_dim]
            # Take the first action
            action = action_chunk[0, 0, :].float().cpu().numpy()
            
            # Step Env
            # Metaworld expects 4 dims: [xyz, gripper]
            step_result = env.step(action)
            
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            if info.get('success', False):
                successes += 1
                print(f"Episode {ep}: SUCCESS at step {step}")
                done = True
                
            step += 1
            if step % 50 == 0: print(f"  Step {step}")
            
        print(f"Episode {ep} finished. Success: {info.get('success', False)}")

        # Save Video for this episode
        if episode_frames:
            video_path = f"eval_metaworld_ep{ep}.mp4"
            height, width, _ = episode_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for frame in episode_frames:
                out.write(frame)
            out.release()
            print(f"Saved evaluation video to {os.path.abspath(video_path)}")

if __name__ == "__main__":
    main()
