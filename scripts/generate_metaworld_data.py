import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import h5py
import argparse
from pathlib import Path
import metaworld
import metaworld.policies
import random
import cv2

def generate_metaworld_episode(env_name, episode_length=500, image_size=(480, 640)):
    # Initialize implementation of specific metaworld env
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name](render_mode='rgb_array')
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
    left_images = []
    right_images = []
    top_images = []
    
    curr_obs = obs
    for t in range(episode_length):
        action = policy.get_action(curr_obs)
        
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
        # With render_mode='rgb_array' set at init, render() typically takes no args
        
        
        
        # img = env.render()
        img = np.zeros((*image_size, 3), dtype=np.uint8)  



        # Resize/Verify size
        # Metaworld render output might be arbitrary size depending on mujoco config
        # We ensure it matches desired 480x640
        if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
             img = cv2.resize(img, (image_size[1], image_size[0]))

        left_images.append(img)
        right_images.append(img) # Placeholder: duplicate
        top_images.append(img)   # Placeholder: duplicate
        
        # Robot state
        # Safe access to qpos
        try:
            full_qpos = env.sim.data.qpos.flat[:].copy()
            full_qvel = env.sim.data.qvel.flat[:].copy()
        except:
            # Fallback for some versions
            full_qpos = np.zeros(7)
            full_qvel = np.zeros(7)

        qpos_list.append(full_qpos)
        qvel_list.append(full_qvel)
        action_list.append(action)
        
        curr_obs = next_obs
        if done or info.get('success', False):
            # Continue for a bit or break?
            # Ideally we want full episode length for batching or handle variable length.
            pass
            
    return {
        'qpos': np.array(qpos_list),
        'qvel': np.array(qvel_list),
        'action': np.array(action_list),
        'left_images': np.array(left_images),
        'right_images': np.array(right_images),
        'top_images': np.array(top_images),
        'language': f"interactions with {env_name}"
    }

def save_episode_hdf5(save_path, episode_data):
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('action', data=episode_data['action'], dtype=np.float32)
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=episode_data['qpos'], dtype=np.float32)
        obs_group.create_dataset('qvel', data=episode_data['qvel'], dtype=np.float32)
        img_group = obs_group.create_group('images')
        img_group.create_dataset('left', data=episode_data['left_images'], dtype=np.uint8)
        img_group.create_dataset('right', data=episode_data['right_images'], dtype=np.uint8)
        img_group.create_dataset('top', data=episode_data['top_images'], dtype=np.uint8)
        f.create_dataset('language_raw', data=[episode_data['language'].encode('utf-8')], dtype=h5py.string_dtype())
        f.attrs['sim'] = True
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='data/metaworld_task')
    parser.add_argument('--num_episodes', type=int, default=50) # Generate 50 episodes
    parser.add_argument('--env_name', type=str, default='pick-place-v3')
    args = parser.parse_args()
    
    # Clean/Create dir
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_episodes} episodes for {args.env_name}...")
    
    import traceback
    for i in range(args.num_episodes):
        if i % 10 == 0: print(f"Episode {i}")
        try:
            data = generate_metaworld_episode(args.env_name)
            save_path = os.path.join(args.output_dir, f"episode_{i}.hdf5")
            save_episode_hdf5(save_path, data)
        except Exception as e:
            print(f"Error in episode {i}: {e}")
            traceback.print_exc()
            
    print("Generation complete.")

if __name__ == "__main__":
    main()
