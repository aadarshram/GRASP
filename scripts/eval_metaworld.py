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
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to trained checkpoint (output_dir)")
    parser.add_argument("--hf_model", type=str, default=None, help="HuggingFace model ID for diffusion head (e.g., hz1919810/TinyVLA-droid_diffusion_metaworld)")
    parser.add_argument("--hf_head_file", type=str, default="diff_head_ft.pth", help="Diffusion head checkpoint filename in HF repo")
    parser.add_argument("--base_model", type=str, default="lesjie/Llava-Pythia-400M")
    parser.add_argument("--env_name", type=str, default="pick-place-v3")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--action_dim", type=int, default=4)
    args = parser.parse_args()
    if not args.ckpt_path and not args.hf_model:
        parser.error("Either --ckpt_path or --hf_model must be provided")
    return args

def main():
    args = get_args()
    
    # 1. Load Config & Model
    if args.hf_model:
        print(f"Loading VLM from {args.base_model}...")
        print(f"Will load diffusion head from HuggingFace: {args.hf_model}/{args.hf_head_file}")
    else:
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
        bf16=False,
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
    
    # Determine base model name
    # For HF models, always use args.base_model (lesjie/Llava-Pythia-400M)
    # For local checkpoints, use args.base_model
    base_model_name = args.base_model
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="right",
        model_max_length=2048
    )
    tokenizer.pad_token_id = 1 # As per training log

    llava_pythia_config = LlavaPythiaConfig.from_pretrained(base_model_name, trust_remote_code=True)
    llava_pythia_config.action_head_type = 'droid_diffusion'
    llava_pythia_config.action_dim = args.action_dim
    llava_pythia_config.state_dim = 7
    llava_pythia_config.chunk_size = 16 # Default in train.py
    llava_pythia_config.concat = "token_cat"

    # Update model_args to use the correct base model
    model_args.model_name_or_path = base_model_name

    # For HF models (diffusion head only), disable LoRA
    # For local checkpoints, enable LoRA
    use_lora = args.ckpt_path is not None
    training_args.lora_enable = use_lora

    # Load Base Model + LoRA init (randomly initialized LoRA layers)
    config = {
        'model_args': model_args,
        'training_args': training_args,
        'data_args': data_args,
        'bnb_model_from_pretrained_args': {}
    }
    
    model, _ = load_llava_pythia(config=config, llava_pythia_config=llava_pythia_config, tokenizer=tokenizer)
    
    # Load weights based on source
    if args.hf_model:
        # HF model contains only the diffusion head checkpoint
        from huggingface_hub import hf_hub_download
        print(f"Downloading diffusion head from HuggingFace: {args.hf_model}/{args.hf_head_file}...")
        
        try:
            head_path = hf_hub_download(repo_id=args.hf_model, filename=args.hf_head_file)
            print(f"Loading diffusion head from {head_path}")
            
            # Load the diffusion head state dict
            head_state_dict = torch.load(head_path, map_location='cpu')
            
            # The checkpoint contains diffusion head and mm_projector weights
            # Filter for action_head and mm_projector keys
            weights_to_load = {}
            for k, v in head_state_dict.items():
                # Remove potential prefixes
                k_clean = k.replace('module.', '').replace('base_model.model.', '')
                # Load both action_head (diffusion head) and mm_projector
                if 'action_head' in k_clean or 'mm_projector' in k_clean or 'embed_out' in k_clean:
                    # Keep as float32
                    weights_to_load[k_clean] = v.float()
            
            if not weights_to_load:
                # If no expected keys, assume entire dict contains the weights
                print("No 'action_head' or 'mm_projector' found, loading entire checkpoint")
                weights_to_load = {}
                for k, v in head_state_dict.items():
                    k_clean = k.replace('module.', '').replace('base_model.model.', '')
                    # Try to infer the structure - if it has model weights, prepend action_head
                    if not k_clean.startswith(('action_head', 'mm_projector', 'embed_out', 'gpt_neox')):
                        # Likely diffusion head weights without prefix
                        # Keep as float32
                        weights_to_load[f'action_head.{k_clean}'] = v.float()
                    else:
                        weights_to_load[k_clean] = v.float()
            
            # Load into model
            missing, unexpected = model.load_state_dict(weights_to_load, strict=False)
            print(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            print(f"Loaded components: {set([k.split('.')[0] for k in weights_to_load.keys()])}")
            if len(unexpected) > 0:
                print(f"Unexpected keys: {unexpected[:5]}...")  # Show first 5
            
        except Exception as e:
            print(f"Error loading diffusion head from HuggingFace: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Local checkpoint - load LoRA and non-LoRA weights
        if args.ckpt_path:
            ckpt_path_abs = os.path.abspath(args.ckpt_path)
            print(f"Loading from local checkpoint: {ckpt_path_abs}...")
            
            # Load non-LoRA weights (projector, action head)
            non_lora_path = os.path.join(ckpt_path_abs, "non_lora_trainables.bin")
            if not os.path.exists(non_lora_path):
                # Check parent directory (common output_dir pattern)
                parent_dir = os.path.dirname(ckpt_path_abs.rstrip('/'))
                non_lora_path = os.path.join(parent_dir, "non_lora_trainables.bin")

            if os.path.exists(non_lora_path):
                print(f"Loading non-LoRA weights from {non_lora_path}...")
                non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                
                # Handle keys potentially prefixed with 'base_model.model.'
                new_state_dict = {}
                for k, v in non_lora_state_dict.items():
                    # Remove 'base_model.model.' if present (standard PEFT artifact)
                    k = k.replace('base_model.model.', '')
                    new_state_dict[k] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
            else:
                print("Warning: non_lora_trainables.bin not found! Model might be untrained.")

            # Load LoRA weights (only for local checkpoints with LoRA)
            if use_lora:
                adapter_config_path = os.path.join(ckpt_path_abs, "adapter_config.json")
                adapter_weights_path = os.path.join(ckpt_path_abs, "adapter_model.bin")
                
                # Try safetensors first, then bin
                if not os.path.exists(adapter_weights_path):
                    adapter_weights_path = os.path.join(ckpt_path_abs, "adapter_model.safetensors")
                
                if os.path.exists(adapter_config_path) and os.path.exists(adapter_weights_path):
                    print(f"Loading LoRA adapter from {ckpt_path_abs}...")
                    
                    try:
                        if adapter_weights_path.endswith('.safetensors'):
                            from safetensors.torch import load_file as load_safetensors
                            adapter_state_dict = load_safetensors(adapter_weights_path)
                        else:
                            adapter_state_dict = torch.load(adapter_weights_path, map_location='cpu')
                        
                        # Clean up keys
                        lora_state_dict = {}
                        for k, v in adapter_state_dict.items():
                            k_clean = k.replace('base_model.model.', '')
                            lora_state_dict[k_clean] = v
                        
                        missing, unexpected = model.load_state_dict(lora_state_dict, strict=False)
                        print(f"Loaded LoRA adapter. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                    except Exception as e:
                        print(f"Error loading LoRA adapter: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"Warning: LoRA adapter files not found in {ckpt_path_abs}")
                    print(f"  - adapter_config.json exists: {os.path.exists(adapter_config_path)}")
                    print(f"  - adapter_model.bin exists: {os.path.exists(os.path.join(ckpt_path_abs, 'adapter_model.bin'))}")
                    print(f"  - adapter_model.safetensors exists: {os.path.exists(os.path.join(ckpt_path_abs, 'adapter_model.safetensors'))}")
    
    model.cuda()
    model.eval()
    
    # Image Processor
    image_processor = CLIPImageProcessor.from_pretrained(base_model_name)
    crop_size = image_processor.crop_size
    
    # 2. Setup Environment
    print(f"Initializing Metaworld env: {args.env_name}")
    ml1 = metaworld.ML1(args.env_name)
    env = ml1.train_classes[args.env_name](render_mode='rgb_array')
    env.set_task(ml1.train_tasks[0])
    
    # Helper function to render from a specific camera (matches training code exactly)
    def render_from_camera(camera_id, image_size=(480, 640)):
        """Render from a specific camera ID."""
        # Store original camera_id
        original_camera_id = env.mujoco_renderer.camera_id
        try:
            # Set camera for this render
            env.mujoco_renderer.camera_id = camera_id
            img = env.render()
            
            # Resize if needed
            if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                img = cv2.resize(img, (image_size[1], image_size[0]))
            return img
        except Exception as e:
            # If camera doesn't exist, fall back to default camera
            print(f"Warning: Could not render from camera {camera_id}, using default. Error: {e}")
            env.mujoco_renderer.camera_id = original_camera_id
            img = env.render()
            if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                img = cv2.resize(img, (image_size[1], image_size[0]))
            return img
        finally:
            # Restore original camera
            env.mujoco_renderer.camera_id = original_camera_id
    
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
            # Render from three different camera views (matching training code exactly)
            # Camera IDs: 1 (left), 2 (right), 0 (top)
            img_left = render_from_camera(1, image_size=(180, 320))
            img_left = cv2.rotate(img_left, cv2.ROTATE_180)
            
            img_right = render_from_camera(2, image_size=(180, 320))
            img_right = cv2.rotate(img_right, cv2.ROTATE_180)
            
            img_top = render_from_camera(0, image_size=(180, 320))
            # No rotation for top view
            
            # Stack images horizontally for video
            img_combined = np.hstack([img_left, img_top, img_right])
            # Convert RGB to BGR for OpenCV
            episode_frames.append(cv2.cvtColor(img_combined, cv2.COLOR_RGB2BGR))

            # Prepare images for model inference
            img_pil_left = Image.fromarray(img_left)
            image_tensor_left = image_processor.preprocess(img_pil_left, return_tensors='pt')['pixel_values'][0]
            
            img_pil_right = Image.fromarray(img_right)
            image_tensor_right = image_processor.preprocess(img_pil_right, return_tensors='pt')['pixel_values'][0]
            
            img_pil_top = Image.fromarray(img_top)
            image_tensor_top = image_processor.preprocess(img_pil_top, return_tensors='pt')['pixel_values'][0]
            
            # Stack images for model (left, right, top) matching training data format
            images = image_tensor_left.unsqueeze(0).cuda().float()
            images_r = image_tensor_right.unsqueeze(0).cuda().float()
            images_top = image_tensor_top.unsqueeze(0).cuda().float()
            
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
            state_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
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
            video_path = f"outputs/eval/eval_metaworld_ep{ep}_all_3_views.mp4"
            height, width, _ = episode_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for frame in episode_frames:
                out.write(frame)
            out.release()
            print(f"Saved evaluation video to {os.path.abspath(video_path)}")

if __name__ == "__main__":
    main()
