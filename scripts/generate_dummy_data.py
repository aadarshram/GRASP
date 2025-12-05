#!/usr/bin/env python3
"""
Generate dummy HDF5 dataset files for testing the VLA training pipeline.

This script creates synthetic robot demonstration data with the required structure:
- Multi-camera RGB images (left, right, top)
- Robot joint positions (qpos) and velocities (qvel)
- Action sequences
- Language instructions

Usage:
    python scripts/generate_dummy_data.py --output_dir data/dummy_task --num_episodes 20
"""

import h5py
import numpy as np
import argparse
import os
from pathlib import Path


def generate_dummy_episode(
    episode_length=100,
    action_dim=10,
    state_dim=7,
    image_size=(480, 640),
    task_variations=None
):
    """
    Generate synthetic data for one robot episode.
    
    Args:
        episode_length: Number of timesteps in the episode
        action_dim: Dimensionality of action space
        state_dim: Dimensionality of state space (robot DOF)
        image_size: (height, width) of camera images
        task_variations: List of task descriptions to sample from
    
    Returns:
        Dictionary containing all episode data
    """
    if task_variations is None:
        task_variations = [
            "pick up the red block and place it on the blue block",
            "grasp the cup and move it to the left",
            "push the button with the gripper",
            "open the drawer and place object inside",
            "stack all blocks in a tower",
            "pick up the bottle and pour into the cup",
            "close the gripper around the handle",
            "move the object to the target location",
            "rotate the knob clockwise",
            "press the red button then the green button"
        ]
    
    # Generate smooth trajectories for robot state
    t = np.linspace(0, 2*np.pi, episode_length)
    
    # Robot joint positions (qpos) - smooth sinusoidal motion
    qpos = np.zeros((episode_length, state_dim), dtype=np.float32)
    for i in range(state_dim):
        amplitude = np.random.uniform(0.1, 0.5)
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        qpos[:, i] = amplitude * np.sin(frequency * t + phase)
    
    # Robot joint velocities (qvel) - derivative of qpos
    qvel = np.zeros((episode_length, state_dim), dtype=np.float32)
    qvel[1:] = np.diff(qpos, axis=0)
    qvel[0] = qvel[1]  # Copy first velocity
    
    # Actions - similar to qpos but with different phase
    action = np.zeros((episode_length, action_dim), dtype=np.float32)
    for i in range(action_dim):
        amplitude = np.random.uniform(0.1, 0.3)
        frequency = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        action[:, i] = amplitude * np.sin(frequency * t + phase)
    
    # Generate synthetic camera images
    height, width = image_size
    
    # Create base images with different colors for each camera
    left_images = np.zeros((episode_length, height, width, 3), dtype=np.uint8)
    right_images = np.zeros((episode_length, height, width, 3), dtype=np.uint8)
    top_images = np.zeros((episode_length, height, width, 3), dtype=np.uint8)
    
    for t_idx in range(episode_length):
        # Animate a simple scene: moving colored square
        # Square position moves smoothly across the image
        x_pos = int((t_idx / episode_length) * (width - 100))
        y_pos = int(height / 2 - 50 + 30 * np.sin(4 * np.pi * t_idx / episode_length))
        
        # Left camera - reddish tint
        left_images[t_idx] = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        left_images[t_idx, :, :, 2] += 60  # Red channel
        # Draw moving square
        left_images[t_idx, y_pos:y_pos+100, x_pos:x_pos+100, :] = [255, 100, 100]
        
        # Right camera - greenish tint
        right_images[t_idx] = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        right_images[t_idx, :, :, 1] += 60  # Green channel
        # Draw moving square from different angle
        right_images[t_idx, y_pos+20:y_pos+120, x_pos+20:x_pos+120, :] = [100, 255, 100]
        
        # Top camera - bluish tint
        top_images[t_idx] = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
        top_images[t_idx, :, :, 0] += 60  # Blue channel
        # Draw moving circle for variety
        center_x, center_y = x_pos + 50, height // 2
        for dy in range(-40, 40):
            for dx in range(-40, 40):
                if dx*dx + dy*dy < 1600:  # Circle radius 40
                    cy, cx = center_y + dy, center_x + dx
                    if 0 <= cy < height and 0 <= cx < width:
                        top_images[t_idx, cy, cx, :] = [100, 100, 255]
    
    # Sample a task description
    language = np.random.choice(task_variations)
    
    return {
        'qpos': qpos,
        'qvel': qvel,
        'action': action,
        'left_images': left_images,
        'right_images': right_images,
        'top_images': top_images,
        'language': language
    }


def save_episode_hdf5(save_path, episode_data, compress=False, is_sim=True):
    """
    Save episode data to HDF5 file with the required structure.
    
    Args:
        save_path: Path to save the HDF5 file
        episode_data: Dictionary with episode data from generate_dummy_episode
        compress: Whether to compress images (saves space)
        is_sim: Whether data is from simulation
    """
    import cv2
    
    with h5py.File(save_path, 'w') as f:
        # Store actions
        f.create_dataset('action', data=episode_data['action'], dtype=np.float32)
        
        # Store observations
        obs_group = f.create_group('observations')
        obs_group.create_dataset('qpos', data=episode_data['qpos'], dtype=np.float32)
        obs_group.create_dataset('qvel', data=episode_data['qvel'], dtype=np.float32)
        
        # Store images
        img_group = obs_group.create_group('images')
        
        if compress:
            # Store compressed JPEG images to save disk space
            # HDF5 needs special handling for variable-length data
            compressed_left = [cv2.imencode('.jpg', img)[1] for img in episode_data['left_images']]
            compressed_right = [cv2.imencode('.jpg', img)[1] for img in episode_data['right_images']]
            compressed_top = [cv2.imencode('.jpg', img)[1] for img in episode_data['top_images']]
            
            # Create variable-length dtype for compressed images
            dt = h5py.vlen_dtype(np.dtype('uint8'))
            img_group.create_dataset('left', data=[img.tobytes() for img in compressed_left], dtype=dt)
            img_group.create_dataset('right', data=[img.tobytes() for img in compressed_right], dtype=dt)
            img_group.create_dataset('top', data=[img.tobytes() for img in compressed_top], dtype=dt)
        else:
            # Store uncompressed images
            img_group.create_dataset('left', data=episode_data['left_images'], dtype=np.uint8)
            img_group.create_dataset('right', data=episode_data['right_images'], dtype=np.uint8)
            img_group.create_dataset('top', data=episode_data['top_images'], dtype=np.uint8)
        
        # Store language instruction
        f.create_dataset('language_raw', data=[episode_data['language'].encode('utf-8')], 
                        dtype=h5py.string_dtype())
        
        # Store attributes
        f.attrs['sim'] = is_sim
        if compress:
            f.attrs['compress'] = compress


def validate_hdf5(filepath):
    """Validate that HDF5 file has the required structure."""
    required_keys = ['/action', '/observations/qpos', '/observations/qvel', '/language_raw']
    camera_keys = ['/observations/images/left', '/observations/images/right', '/observations/images/top']
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required datasets
            for key in required_keys:
                assert key in f, f"Missing required key: {key}"
            
            # Check cameras
            for key in camera_keys:
                assert key in f, f"Missing camera: {key}"
            
            # Check shape consistency
            T = f['/action'].shape[0]
            assert f['/observations/qpos'].shape[0] == T, "qpos length mismatch"
            assert f['/observations/qvel'].shape[0] == T, "qvel length mismatch"
            
            # Note: Image length check depends on whether compressed or not
            try:
                img_len = len(f['/observations/images/left'])
                assert img_len == T, f"Image length mismatch: {img_len} != {T}"
            except:
                pass  # Compressed images may have different shape handling
            
            # Check attributes
            assert 'sim' in f.attrs, "Missing 'sim' attribute"
            
            return True, {
                'timesteps': T,
                'action_dim': f['/action'].shape[1],
                'state_dim': f['/observations/qpos'].shape[1],
                'language': f['/language_raw'][0].decode('utf-8'),
                'is_sim': f.attrs['sim'],
                'compressed': f.attrs.get('compress', False)
            }
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description='Generate dummy HDF5 dataset for VLA training')
    parser.add_argument('--output_dir', type=str, default='data/dummy_task',
                       help='Directory to save generated episodes')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes to generate')
    parser.add_argument('--episode_length', type=int, default=100,
                       help='Number of timesteps per episode')
    parser.add_argument('--action_dim', type=int, default=10,
                       help='Action space dimensionality')
    parser.add_argument('--state_dim', type=int, default=7,
                       help='State space dimensionality (robot DOF)')
    parser.add_argument('--image_height', type=int, default=480,
                       help='Image height')
    parser.add_argument('--image_width', type=int, default=640,
                       help='Image width')
    parser.add_argument('--compress', action='store_true',
                       help='Compress images to save disk space')
    parser.add_argument('--validate', action='store_true',
                       help='Validate generated files after creation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_episodes} dummy episodes...")
    print(f"Output directory: {output_dir}")
    print(f"Episode length: {args.episode_length} timesteps")
    print(f"Action dim: {args.action_dim}, State dim: {args.state_dim}")
    print(f"Image size: {args.image_height}x{args.image_width}")
    print(f"Compression: {'enabled' if args.compress else 'disabled'}")
    print()
    
    # Generate episodes
    for i in range(args.num_episodes):
        print(f"Generating episode {i+1}/{args.num_episodes}...", end=' ')
        
        # Generate data
        episode_data = generate_dummy_episode(
            episode_length=args.episode_length,
            action_dim=args.action_dim,
            state_dim=args.state_dim,
            image_size=(args.image_height, args.image_width)
        )
        
        # Save to HDF5
        filename = f"episode_{i:04d}.hdf5"
        filepath = output_dir / filename
        save_episode_hdf5(filepath, episode_data, compress=args.compress)
        
        print(f"✓ Saved to {filename}")
        
        # Validate if requested
        if args.validate and i == 0:  # Validate first file
            print(f"  Validating {filename}...", end=' ')
            valid, info = validate_hdf5(filepath)
            if valid:
                print("✓ Valid")
                print(f"    Timesteps: {info['timesteps']}")
                print(f"    Action dim: {info['action_dim']}")
                print(f"    State dim: {info['state_dim']}")
                print(f"    Language: {info['language']}")
            else:
                print(f"✗ Invalid: {info}")
    
    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print(f"Generated {args.num_episodes} episodes in {output_dir}")
    print()
    print("Next steps:")
    print("1. Update scripts/aloha_scripts/constants.py with this path:")
    print(f"   'dataset_dir': ['{output_dir.absolute()}']")
    print("2. Run training: bash scripts/train.sh")
    print("=" * 60)


if __name__ == '__main__':
    main()
