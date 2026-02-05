
import numpy as np
import h5py
import argparse
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import metaworld
import metaworld.policies
import random
import cv2
import mujoco
import multiprocessing
from functools import partial

def look_at_quat(pos, target, up=np.array([0, 0, 1])):
    z_axis = pos - target
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.vstack((x_axis, y_axis, z_axis)).T
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return np.array([qw, qx, qy, qz])

def generate_metaworld_episode(env_name, episode_length=500, image_size=(480, 640), early_stop=False, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Initialize implementation of specific metaworld env
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name](render_mode='rgb_array')
    # Force use of EGL if available for faster headless rendering in some setups, 
    # but usually handled by env var MUJOCO_GL=egl

    task = random.choice(ml1.train_tasks)
    env.set_task(task)
    
    # Get scripted policy
    if env_name == 'pick-place-v3':
        policy = metaworld.policies.SawyerPickPlaceV3Policy()
    else:
        raise ValueError(f"Add policy mapping for {env_name}")

    obs = env.reset()
    # Handle Gymnasium API (obs, info)
    if isinstance(obs, tuple):
        obs = obs[0]
    
    # buffers
    qpos_list = []
    qvel_list = []
    action_list = []
    top_images = []
    front_images = []
    
    curr_obs = obs
    for t in range(episode_length):
        action = policy.get_action(curr_obs)
        action = np.clip(action, -1.0, 1.0) # Clip action to valid range to prevent exploding gradients
        
        # Step
        step_result = env.step(action)
        # Handle Gymnasium API (obs, reward, terminated, truncated, info) or Old (obs, reward, done, info)
        if len(step_result) == 5:
             next_obs, reward, terminated, truncated, info = step_result
             done = terminated or truncated
        else:
             next_obs, reward, done, info = step_result

        # Handle Gymnasium API obs if needed (though step unpacks it)
        # If the environment wraps obs, next_obs is already the obs.
        
        # Capture images
        # Camera names in Metaworld/Sawyer environments
        # Usually: 'corner', 'topview', 'behindGripper', 'corner2', 'corner3'
        
        # Helper to render specific camera
        def render_camera(cam_name):
            try:
                # Get ID
                cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id == -1:
                    # Try fallback for 'corner2' -> 'behindGripper' etc if needed, or raise
                    if cam_name == 'corner2' or cam_name == 'front':
                        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, 'corner2')
                
                if cam_id != -1:
                    # Custom Front View Logic
                    if cam_name == 'front':
                         # Front Plus Y configuration
                         pos = np.array([0.0, 1.0, 0.6])
                         target = np.array([0.0, 0.5, 0.0])
                         env.model.cam_pos[cam_id] = pos
                         env.model.cam_quat[cam_id] = look_at_quat(pos, target)
                         mujoco.mj_forward(env.model, env.data)

                    env.mujoco_renderer.camera_id = cam_id
                    env.mujoco_renderer.camera_name = None # Ensure ID is used
                    img = env.render()
                    img = cv2.flip(img, 0) # Flip vertically (Mujoco default is upside down relative to OpenCV)
                    
                    # Fix for custom front view being upside down
                    if cam_name == 'front':
                        img = cv2.rotate(img, cv2.ROTATE_180)
                        
                    return img
            except Exception as e:
                print(f"Warning: Could not render {cam_name}: {e}")
            return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # Top View
        top_img = render_camera('topview')
        if top_img.shape[0] != image_size[0] or top_img.shape[1] != image_size[1]:
             top_img = cv2.resize(top_img, (image_size[1], image_size[0]))
        top_images.append(top_img.copy())

        # Front View (using corner2 ID but overriding pose)
        front_img = render_camera('front')
        if front_img.shape[0] != image_size[0] or front_img.shape[1] != image_size[1]:
             front_img = cv2.resize(front_img, (image_size[1], image_size[0]))
        front_images.append(front_img.copy())
        
        # Robot state
        # Safe access to qpos
        try:
            full_qpos = env.data.qpos.flat[:].copy()
            full_qvel = env.data.qvel.flat[:].copy()
        except:
            # Fallback for some versions
            full_qpos = np.zeros(7)
            full_qvel = np.zeros(7)

        qpos_list.append(full_qpos)
        qvel_list.append(full_qvel)
        action_list.append(action)
        
        curr_obs = next_obs
        curr_obs = next_obs
        if done or info.get('success', False):
            if early_stop and info.get('success', False):
                 print(f"Success at step {t}")
                 break
            # Continue for a bit or break?
            # Ideally we want full episode length for batching or handle variable length.
            pass
            
    return {
        'qpos': np.array(qpos_list),
        'qvel': np.array(qvel_list),
        'action': np.array(action_list),
        'top_images': np.array(top_images),
        'front_images': np.array(front_images),
        'language': f"interactions with {env_name}"
    }

def save_episode_hdf5(save_path, episode_data):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('action', data=episode_data['action'], dtype=np.float32)
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=episode_data['qpos'], dtype=np.float32)
        obs_group.create_dataset('qvel', data=episode_data['qvel'], dtype=np.float32)
        img_group = obs_group.create_group('images')
        img_group.create_dataset('top', data=episode_data['top_images'], dtype=np.uint8, compression='gzip')
        img_group.create_dataset('front', data=episode_data['front_images'], dtype=np.uint8, compression='gzip')
        f.create_dataset('language_raw', data=[episode_data['language'].encode('utf-8')], dtype=h5py.string_dtype())
        f.attrs['sim'] = True
        
def worker(args_tuple):
    i, env_name, output_dir, early_stop = args_tuple
    try:
        # Use a unique seed based on the index
        seed = i * 1000 + random.randint(0, 1000)
        data = generate_metaworld_episode(env_name, early_stop=early_stop, seed=seed)
        save_path = os.path.join(output_dir, f"episode_{i}.hdf5")
        save_episode_hdf5(save_path, data)
        if i % 10 == 0:
             print(f"Generated episode {i}")
        return True
    except Exception as e:
        print(f"Error in episode {i}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/metaworld_task')
    parser.add_argument('--num_episodes', type=int, default=50) 
    parser.add_argument('--env_name', type=str, default='pick-place-v3')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--early_stop', action='store_true', help='Stop episode on success')
    args = parser.parse_args()
    
    # Clean/Create dir
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_episodes} episodes for {args.env_name} with {args.num_workers} workers...")
    
    # Prepare arguments for workers
    worker_args = [(i, args.env_name, args.output_dir, args.early_stop) for i in range(args.num_episodes)]
    
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pool.map(worker, worker_args)
    else:
        for arg in worker_args:
            worker(arg)
            
    print("Generation complete.")

if __name__ == "__main__":
    main()
