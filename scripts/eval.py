"""
Evaluation script for Vision-Language-Action models on Metaworld and Libero environments.

This script supports evaluation of:
1. Local checkpoints (with LoRA adapters + non-LoRA weights)
2. HuggingFace models (diffusion head only)

on both Metaworld and Libero benchmarks.

Example Usage:
--------------

1. Evaluate local checkpoint on Metaworld:
   python scripts/eval.py \
       --env_type metaworld \
       --env_name pick-place-v3 \
       --ckpt_path outputs/metaworld_train/checkpoint-4000 \
       --action_dim 4 \
       --state_dim 7 \
       --num_episodes 5 \
       --output_dir outputs/eval/sachin_model

2. Evaluate local checkpoint on Libero:
   python scripts/eval.py \
       --env_type libero \
       --task_suite_name libero_spatial \
       --task_id 0 \
       --ckpt_path outputs/metaworld_train/checkpoint-4000 \
       --action_dim 7 \
       --state_dim 7 \
       --num_episodes 5 \
       --output_dir outputs/eval/sachin_model

3. Evaluate HuggingFace model on Metaworld:
   python scripts/eval.py \
       --env_type metaworld \
       --env_name pick-place-v3 \
       --hf_model hz1919810/TinyVLA-droid_diffusion_metaworld \
       --hf_head_file diff_head_raw_final.pth \
       --action_dim 4 \
       --state_dim 7 \
       --num_episodes 5 \
       --output_dir outputs/eval/HF_model

4. Evaluate HuggingFace model on Libero:
   python scripts/eval.py \
       --env_type libero \
       --task_suite_name libero_spatial \
       --task_id 0 \
       --hf_model hz1919810/TinyVLA-droid_diffusion_metaworld \
       --hf_head_file diff_head_raw_final.pth \
       --action_dim 7 \
       --state_dim 7 \
       --num_episodes 5 \
       --output_dir outputs/eval/HF_model

Output:
-------
- Evaluation videos are saved to: outputs/eval/eval_{env_type}_{task_name}_ep{episode_id}.mp4
- Videos show three camera views stacked horizontally: [left, top, right]
- metaworld cam ids: { 0: top, 1: left, 2: right, 3: top-right, 4: front, 5: gripper}
- libero camera names: agentview, birdview, sideview, robot0_eye_in_hand, frontview, galleryview, robot0_robotview
- Console outputs success rate and episode statistics
"""

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

try:
    import libero
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("Warning: libero not available. Install it to use libero environments.")

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
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate VLA models on Metaworld or Libero environments")
    
    # Model source (mutually exclusive: local checkpoint OR HuggingFace model)
    parser.add_argument("--ckpt_path", type=str, default=None, 
                        help="Path to local trained checkpoint directory (contains adapter_model.bin and non_lora_trainables.bin)")
    parser.add_argument("--hf_model", type=str, default=None, 
                        help="HuggingFace model ID for diffusion head (e.g., hz1919810/TinyVLA-droid_diffusion_metaworld)")
    parser.add_argument("--hf_head_file", type=str, default="diff_head_raw_final.pth", 
                        help="Diffusion head checkpoint filename in HF repo")
    parser.add_argument("--base_model", type=str, default="lesjie/Llava-Pythia-400M",
                        help="Base VLM model (LLaVA-Pythia)")
    
    # Environment configuration
    parser.add_argument("--env_type", type=str, default="metaworld", choices=["metaworld", "libero"], 
                        help="Environment type to evaluate on")
    
    # Metaworld-specific parameters
    parser.add_argument("--env_name", type=str, default="pick-place-v3", 
                        help="Metaworld environment name (e.g., pick-place-v3, drawer-open-v3)")
    
    # Libero-specific parameters
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial", 
                        help="Libero task suite (libero_spatial, libero_object, libero_goal, libero_10, libero_90)")
    parser.add_argument("--task_name", type=str, default=None, 
                        help="Specific libero task name (if None, uses task_id)")
    parser.add_argument("--task_id", type=int, default=0, 
                        help="Task index within the suite (used if task_name not specified)")
    
    # Evaluation parameters
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--action_dim", type=int, default=4, 
                        help="Action dimension (4 for metaworld, 7 for libero)")
    parser.add_argument("--state_dim", type=int, default=7, 
                        help="Robot state/proprioception dimension")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                        help="Directory to save evaluation videos")
    
    args = parser.parse_args()
    if not args.ckpt_path and not args.hf_model:
        parser.error("Either --ckpt_path or --hf_model must be provided")
    if args.env_type == "libero" and not LIBERO_AVAILABLE:
        parser.error("libero is not installed. Please install it to use libero environments.")
    return args

def main():
    args = get_args()
    
    # ============================================================================
    # 1. MODEL LOADING
    # ============================================================================
    if args.hf_model:
        print(f"Loading VLM from {args.base_model}...")
        print(f"Will load diffusion head from HuggingFace: {args.hf_model}/{args.hf_head_file}")
    else:
        print(f"Loading model from {args.ckpt_path}...")
    
    # Reconstruct training config structure for load_llava_pythia
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
    
    # Initialize tokenizer
    base_model_name = args.base_model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="right",
        model_max_length=2048
    )
    tokenizer.pad_token_id = 1

    # Configure action head
    llava_pythia_config = LlavaPythiaConfig.from_pretrained(base_model_name, trust_remote_code=True)
    llava_pythia_config.action_head_type = 'droid_diffusion'
    llava_pythia_config.action_dim = args.action_dim
    llava_pythia_config.state_dim = args.state_dim
    llava_pythia_config.chunk_size = 16
    llava_pythia_config.concat = "token_cat"

    model_args.model_name_or_path = base_model_name
    use_lora = args.ckpt_path is not None  # LoRA only for local checkpoints
    training_args.lora_enable = use_lora

    # Load base model with LoRA initialized (will be overwritten by checkpoint weights)
    config = {
        'model_args': model_args,
        'training_args': training_args,
        'data_args': data_args,
        'bnb_model_from_pretrained_args': {}
    }
    
    model, _ = load_llava_pythia(config=config, llava_pythia_config=llava_pythia_config, tokenizer=tokenizer)
    
    # Load checkpoint weights
    if args.hf_model:
        # Load diffusion head from HuggingFace (action_head + mm_projector only)
        from huggingface_hub import hf_hub_download
        print(f"Downloading diffusion head from HuggingFace: {args.hf_model}/{args.hf_head_file}...")
        
        try:
            head_path = hf_hub_download(repo_id=args.hf_model, filename=args.hf_head_file)
            print(f"Loading diffusion head from {head_path}")
            
            head_state_dict = torch.load(head_path, map_location='cpu')
            
            # Extract relevant weights (action_head, mm_projector, embed_out)
            weights_to_load = {}
            for k, v in head_state_dict.items():
                k_clean = k.replace('module.', '').replace('base_model.model.', '')
                if 'action_head' in k_clean or 'mm_projector' in k_clean or 'embed_out' in k_clean:
                    weights_to_load[k_clean] = v.float()
            
            # Fallback: if no expected keys found, infer structure
            if not weights_to_load:
                print("No 'action_head' or 'mm_projector' found, loading entire checkpoint")
                for k, v in head_state_dict.items():
                    k_clean = k.replace('module.', '').replace('base_model.model.', '')
                    if not k_clean.startswith(('action_head', 'mm_projector', 'embed_out', 'gpt_neox')):
                        weights_to_load[f'action_head.{k_clean}'] = v.float()
                    else:
                        weights_to_load[k_clean] = v.float()
            
            missing, unexpected = model.load_state_dict(weights_to_load, strict=False)
            print(f"Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            print(f"Loaded components: {set([k.split('.')[0] for k in weights_to_load.keys()])}")
            
        except Exception as e:
            print(f"Error loading diffusion head from HuggingFace: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        # Load from local checkpoint (LoRA adapters + non-LoRA weights)
        if args.ckpt_path:
            ckpt_path_abs = os.path.abspath(args.ckpt_path)
            print(f"Loading from local checkpoint: {ckpt_path_abs}...")
            
            # 1. Load non-LoRA weights (mm_projector, action_head)
            non_lora_path = os.path.join(ckpt_path_abs, "non_lora_trainables.bin")
            if not os.path.exists(non_lora_path):
                parent_dir = os.path.dirname(ckpt_path_abs.rstrip('/'))
                non_lora_path = os.path.join(parent_dir, "non_lora_trainables.bin")

            if os.path.exists(non_lora_path):
                print(f"Loading non-LoRA weights from {non_lora_path}...")
                non_lora_state_dict = torch.load(non_lora_path, map_location='cpu')
                
                # Clean up PEFT key prefixes
                new_state_dict = {}
                for k, v in non_lora_state_dict.items():
                    k = k.replace('base_model.model.', '')
                    new_state_dict[k] = v
                    
                model.load_state_dict(new_state_dict, strict=False)
            else:
                print("Warning: non_lora_trainables.bin not found! Model might be untrained.")

            # 2. Load LoRA adapter weights
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
    
    # Initialize image processor
    image_processor = CLIPImageProcessor.from_pretrained(base_model_name)
    crop_size = image_processor.crop_size
    
    # ============================================================================
    # 2. ENVIRONMENT SETUP
    # ============================================================================
    print(f"Initializing {args.env_type} environment...")
    task_name = None
    
    if args.env_type == "metaworld":
        print(f"Environment: {args.env_name}")
        ml1 = metaworld.ML1(args.env_name)
        env = ml1.train_classes[args.env_name](render_mode='rgb_array')
        env.set_task(ml1.train_tasks[0])
        task_description = f"interactions with {args.env_name}"
        task_name = args.env_name  # Set for video naming
        
    elif args.env_type == "libero":
        # Load Libero benchmark and task
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        
        if args.task_name:
            task_names = task_suite.get_task_names()
            try:
                task_id = task_names.index(args.task_name)
            except ValueError:
                print(f"Error: Task '{args.task_name}' not found in {args.task_suite_name}")
                print(f"Available tasks: {task_names}")
                sys.exit(1)
        else:
            task_id = args.task_id
        
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        
        print(f"Task Suite: {args.task_suite_name}")
        print(f"Task ID: {task_id}")
        print(f"Task: {task_name}")
        print(f"Description: {task_description}")
        
        # Construct BDDL file path
        try:
            bddl_folder = get_libero_path("bddl_files")
            bddl_file_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        except:
            bddl_file_path = os.path.join(task.problem_folder, task.bddl_file)
        
        # Initialize environment
        env_args = {
            "bddl_file_name": bddl_file_path,
            "camera_heights": 180,
            "camera_widths": 320,
        }
        
        print(f"Initializing environment with BDDL file: {bddl_file_path}")
        env = OffScreenRenderEnv(**env_args)
        env.seed(0)
        obs = env.reset()
        
        # Stabilize physics
        for _ in range(5):
            env.step(np.zeros(args.action_dim))
    
    def render_from_camera(camera_id=None, camera_name=None, image_size=(180, 320)):
        """
        Render from a specific camera.
        
        Args:
            camera_id: Camera index for Metaworld
            camera_name: Camera name for Libero (overrides camera_id if provided)
            image_size: Tuple of (height, width)
            
        Returns:
            RGB image array
        """
        if args.env_type == "libero":
            # Use camera_name if provided, otherwise fall back to index-based lookup
            if camera_name is None:
                camera_names = ["agentview", "robot0_eye_in_hand", "birdview"]
                if camera_id < len(camera_names):
                    camera_name = camera_names[camera_id]
                else:
                    camera_name = "agentview"
            
            try:
                img = env.sim.render(
                    height=image_size[0], 
                    width=image_size[1], 
                    camera_name=camera_name
                )
                img = img[::-1]  # Flip vertically (MuJoCo convention)
                return img
            except Exception as e:
                print(f"Warning: Could not render from camera {camera_name}: {e}")
                try:
                    img = env.sim.render(
                        height=image_size[0], 
                        width=image_size[1], 
                        camera_name="agentview"
                    )
                    img = img[::-1]
                    return img
                except:
                    return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        else:
            # Metaworld: switch camera ID, render, then restore
            original_camera_id = env.mujoco_renderer.camera_id
            try:
                env.mujoco_renderer.camera_id = camera_id
                img = env.render()
                if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                    img = cv2.resize(img, (image_size[1], image_size[0]))
                return img
            except Exception as e:
                print(f"Warning: Could not render from camera {camera_id}, using default. Error: {e}")
                env.mujoco_renderer.camera_id = original_camera_id
                img = env.render()
                if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                    img = cv2.resize(img, (image_size[1], image_size[0]))
                return img
            finally:
                env.mujoco_renderer.camera_id = original_camera_id
    
    # ============================================================================
    # 3. EVALUATION LOOP
    # ============================================================================
    successes = 0
    frames = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple): 
            obs = obs[0]
        
        episode_frames = []
        step = 0
        done = False
        
        while not done and step < 500:
            # Render 4-camera views for both environments
            if args.env_type == "metaworld":
                # METAWORLD cam ids: { 0: top, 1: left, 2: right, 3: top-right, 4: front, 5: gripper}
                img_bottom_left = cv2.rotate(render_from_camera(camera_id=1, image_size=(180, 320)), cv2.ROTATE_180)
                img_bottom_right = cv2.rotate(render_from_camera(camera_id=2, image_size=(180, 320)), cv2.ROTATE_180)
                img_top_left = render_from_camera(camera_id=0, image_size=(180, 320))
                img_top_right = render_from_camera(camera_id=4, image_size=(180, 320))
            else:
                # LIBERO camera names: agentview, birdview, sideview, robot0_eye_in_hand, frontview, galleryview, robot0_robotview
                img_top_right = render_from_camera(camera_name="frontview", image_size=(180, 320))
                img_top_left = render_from_camera(camera_name="birdview", image_size=(180, 320))
                img_bottom_left = render_from_camera(camera_name="sideview", image_size=(180, 320))
                img_bottom_right = render_from_camera(camera_name="agentview", image_size=(180, 320))
            
            top_row = np.hstack([img_top_left, img_top_right])
            bottom_row = np.hstack([img_bottom_left, img_bottom_right])
            img_combined = np.vstack([top_row, bottom_row])
            episode_frames.append(cv2.cvtColor(img_combined, cv2.COLOR_RGB2BGR))

            # Preprocess images for VLM
            image_tensor_left = image_processor.preprocess(Image.fromarray(img_bottom_left), return_tensors='pt')['pixel_values'][0]
            image_tensor_right = image_processor.preprocess(Image.fromarray(img_bottom_right), return_tensors='pt')['pixel_values'][0]
            image_tensor_top = image_processor.preprocess(Image.fromarray(img_top_left), return_tensors='pt')['pixel_values'][0]
            
            images = image_tensor_left.unsqueeze(0).cuda().float()
            images_r = image_tensor_right.unsqueeze(0).cuda().float()
            images_top = image_tensor_top.unsqueeze(0).cuda().float()
            
            # Prepare text prompt
            prompt = task_description + "\n"
            context = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            input_ids = tokenizer_image_token(context, tokenizer, return_tensors='pt').unsqueeze(0).cuda()
            
            # Get robot proprioception (joint positions)
            try:
                if args.env_type == "metaworld":
                    qpos = env.sim.data.qpos.flat[:].copy()[:args.state_dim]
                else:
                    qpos = env.sim.data.qpos[:args.state_dim].copy()
            except:
                qpos = np.zeros(args.state_dim)
            state_tensor = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            # Run model inference
            with torch.no_grad():
                action_chunk = model(
                    input_ids=input_ids,
                    images=images,
                    images_r=images_r,
                    images_top=images_top,
                    states=state_tensor,
                    eval=True
                )
            
            # Extract first action from chunk and clip to valid range
            action = action_chunk[0, 0, :].float().cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            
            # Execute action in environment
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
            if step % 50 == 0: 
                print(f"  Step {step}")
            
        print(f"Episode {ep} finished. Success: {info.get('success', False)}")

        # Save episode video
        if episode_frames:
            os.makedirs(args.output_dir, exist_ok=True)
            video_path = os.path.join(args.output_dir, f"eval_{args.env_type}_{task_name}_ep{ep}.mp4")
            height, width, _ = episode_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            for frame in episode_frames:
                out.write(frame)
            out.release()
            print(f"Saved evaluation video to {os.path.abspath(video_path)}")

if __name__ == "__main__":
    main()