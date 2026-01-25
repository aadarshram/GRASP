
import numpy as np
import h5py
import argparse
import os
from pathlib import Path
import metaworld
import metaworld.policies
import random
import cv2
import mujoco
import multiprocessing
from functools import partial
import sys

# Add scripts directory to path for camera_config import
sys.path.insert(0, os.path.dirname(__file__))
from camera_config import get_camera_names, get_camera_mujoco_names

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

def generate_metaworld_episode(env_name, episode_length=500, image_size=(480, 640), early_stop=False, seed=None, save_video=False):
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
    
    # Get selected cameras from config
    selected_cameras = get_camera_names()
    print(f"Recording with cameras: {selected_cameras}")
    
    # Initialize buffers for selected cameras
    qpos_list = []
    qvel_list = []
    action_list = []
    image_buffers = {cam: [] for cam in selected_cameras}  # Dynamic camera buffers
    video_frames = [] if save_video else None  # Only collect frames if saving video
    
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
        
        # Capture images from selected cameras

        # - Camera ID 1 (left): 'corner'
        # - Camera ID 2 (right): 'corner2'
        # - Camera ID 4 (front): custom position (front-facing view)
        # - Camera ID 5 (gripper): 'behindGripper' (wrist/end-effector view)
        
        # Helper to render specific camera
        def render_camera(cam_name, mujoco_cam_name=None):
            """
            Render a camera view.
            
            Args:
                cam_name: Logical name ('top', 'left', 'right', 'front', 'gripper')
                mujoco_cam_name: Actual MuJoCo camera name in the model
            """
            try:
                # Map logical names to MuJoCo camera names
                mujoco_name_map = {
                    'top': 'topview',
                    'left': 'corner',
                    'right': 'corner2',
                    'front': 'corner2',  # Will override position for front view
                    'gripper': 'behindGripper'
                }
                
                if mujoco_cam_name is None:
                    mujoco_cam_name = mujoco_name_map.get(cam_name, cam_name)
                
                # Get camera ID from MuJoCo model
                cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, mujoco_cam_name)
                
                if cam_id != -1:
                    # Custom Front View: Override camera position for front-facing view
                    if cam_name == 'front':
                         # Front view configuration (looking at workspace from front)
                         pos = np.array([0.0, 1.0, 0.6])
                         target = np.array([0.0, 0.5, 0.0])
                         env.model.cam_pos[cam_id] = pos
                         env.model.cam_quat[cam_id] = look_at_quat(pos, target)
                         mujoco.mj_forward(env.model, env.data)

                    env.mujoco_renderer.camera_id = cam_id
                    env.mujoco_renderer.camera_name = None # Ensure ID is used
                    img = env.render()
                    img = cv2.flip(img, 0) # Flip vertically (MuJoCo default is upside down)
                    
                    # Fix for custom front view orientation
                    if cam_name == 'front':
                        img = cv2.rotate(img, cv2.ROTATE_180)
                        
                    return img
                else:
                    print(f"Warning: Camera '{mujoco_cam_name}' not found in environment")
            except Exception as e:
                print(f"Warning: Could not render {cam_name} ({mujoco_cam_name}): {e}")
            return np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

        # Render selected cameras dynamically
        for cam_name in selected_cameras:
            img = render_camera(cam_name)
            if img.shape[0] != image_size[0] or img.shape[1] != image_size[1]:
                img = cv2.resize(img, (image_size[1], image_size[0]))
            image_buffers[cam_name].append(img.copy())
        
        # If saving video, stack cameras for the frame
        if save_video:
            # Stack selected cameras for visualization
            camera_images = {cam: image_buffers[cam][-1] for cam in selected_cameras}
            num_cams = len(selected_cameras)
            if num_cams == 1:
                frame = camera_images[selected_cameras[0]]
            elif num_cams == 2:
                frame = np.hstack([camera_images[selected_cameras[0]], camera_images[selected_cameras[1]]])
            elif num_cams == 3:
                frame = np.hstack([camera_images[selected_cameras[0]], camera_images[selected_cameras[1]], camera_images[selected_cameras[2]]])
            elif num_cams >= 4:
                row1 = np.hstack([camera_images[selected_cameras[0]], camera_images[selected_cameras[1]]])
                row2 = np.hstack([camera_images[selected_cameras[2]], camera_images[selected_cameras[3]]])
                frame = np.vstack([row1, row2])
            else:
                frame = camera_images[selected_cameras[0]]
            # Convert RGB to BGR for OpenCV video writing
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_frames.append(frame_bgr)
        
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
            
    # Build return dict with selected cameras only
    return_dict = {
        'qpos': np.array(qpos_list),
        'qvel': np.array(qvel_list),
        'action': np.array(action_list),
        'language': f"interactions with {env_name}",
        'selected_cameras': selected_cameras
    }
    # Add image buffers for selected cameras
    for cam_name in selected_cameras:
        return_dict[f'{cam_name}_images'] = np.array(image_buffers[cam_name])
    
    # Add video frames if collected
    if save_video and video_frames:
        return_dict['video_frames'] = video_frames
    
    return return_dict

def save_episode_hdf5(save_path, episode_data):
    """
    Save episode data to HDF5 file with selected cameras.
    
    The HDF5 structure dynamically includes only the selected cameras from camera_config.
    """
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('action', data=episode_data['action'], dtype=np.float32)
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=episode_data['qpos'], dtype=np.float32)
        obs_group.create_dataset('qvel', data=episode_data['qvel'], dtype=np.float32)
        
        # Create images group with only selected cameras
        img_group = obs_group.create_group('images')
        selected_cameras = episode_data.get('selected_cameras', get_camera_names())
        for cam_name in selected_cameras:
            img_key = f'{cam_name}_images'
            if img_key in episode_data:
                img_group.create_dataset(cam_name, data=episode_data[img_key], dtype=np.uint8, compression='gzip')
        
        f.create_dataset('language_raw', data=[episode_data['language'].encode('utf-8')], dtype=h5py.string_dtype())
        f.attrs['sim'] = True


def save_episode_video(save_path, video_frames, fps=30):
    """
    Save episode video frames to MP4 file.
    
    Args:
        save_path: Path to save the MP4 video file
        video_frames: List of frames (BGR numpy arrays)
        fps: Frames per second for the video
    """
    if not video_frames:
        print(f"Warning: No frames to save for video at {save_path}")
        return
    
    # Get frame dimensions
    frame_height, frame_width = video_frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    # Write frames to video
    for frame in video_frames:
        out.write(frame)
    
    out.release()
    print(f"Saved video to {save_path}")

        
def worker(args_tuple):
    i, env_name, output_dir, early_stop, save_video = args_tuple
    try:
        # Use a unique seed based on the index
        seed = i * 1000 + random.randint(0, 1000)
        data = generate_metaworld_episode(env_name, early_stop=early_stop, seed=seed, save_video=save_video)
        
        # Save HDF5 file
        hdf5_path = os.path.join(output_dir, f"episode_{i}.hdf5")
        save_episode_hdf5(hdf5_path, data)
        
        # Save video if requested and frames are available
        if save_video and 'video_frames' in data:
            videos_dir = output_dir
            os.makedirs(videos_dir, exist_ok=True)
            video_path = os.path.join(videos_dir, f"episode_{i}.mp4")
            save_episode_video(video_path, data['video_frames'])
        
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
    parser.add_argument('--output_dir', type=str, default='data/metaworld_pick-place-v3')
    parser.add_argument('--num_episodes', type=int, default=5) 
    parser.add_argument('--env_name', type=str, default='pick-place-v3')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--early_stop', action='store_true', help='Stop episode on success')
    parser.add_argument('--save_video', action='store_true', help='Save MP4 videos in addition to HDF5 files')
    args = parser.parse_args()
    
    # Clean/Create dir
    if os.path.exists(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.num_episodes} episodes for {args.env_name} with {args.num_workers} workers...")
    print(f"Save video: {args.save_video}")
    
    # Prepare arguments for workers
    worker_args = [(i, args.env_name, args.output_dir, args.early_stop, args.save_video) for i in range(args.num_episodes)]
    
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pool.map(worker, worker_args)
    else:
        for arg in worker_args:
            worker(arg)
            
    print("Generation complete.")
    if args.save_video:
        print(f"Videos saved to: {os.path.join(args.output_dir, 'videos')}")

if __name__ == "__main__":
    main()
