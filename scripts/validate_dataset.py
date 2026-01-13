#!/usr/bin/env python3
"""
Validate HDF5 dataset files for VLA training.

This script checks that all HDF5 files in a directory have the required structure
and reports any issues.

Usage:
    python scripts/validate_dataset.py --data_dir data/dummy_task
"""

import h5py
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add scripts directory to path for camera_config import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
from camera_config import get_camera_names, METAWORLD_CAMERAS


def validate_hdf5_file(filepath, verbose=False):
    """
    Validate that an HDF5 file has the required structure for VLA training.
    
    This validates the structure required by train.py, including:
    - Action sequences (action vectors for all timesteps)
    - Robot state (qpos and qvel for proprioception)
    - Multi-camera images (at least one camera required)
    - Language instructions (task description)
    - Dataset metadata (sim attribute)
    
    Selected cameras from camera_config.py are validated.
    
    Args:
        filepath: Path to HDF5 file
        verbose: Print detailed information
    
    Returns:
        (is_valid, info_dict, errors_list)
    """
    # Core required datasets for train.py
    required_keys = ['/action', '/observations/qpos', '/observations/qvel', '/language_raw']
    
    # Get selected cameras from camera_config
    selected_cameras = get_camera_names()
    # All available cameras for reference
    all_possible_cameras = list(METAWORLD_CAMERAS.keys())
    
    errors = []
    info = {}
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required datasets
            for key in required_keys:
                if key not in f:
                    errors.append(f"Missing required dataset: {key}")
            
            # Check that selected cameras exist
            available_cameras = []
            missing_cameras = []
            for cam in selected_cameras:
                if f'/observations/images/{cam}' in f:
                    available_cameras.append(cam)
                else:
                    missing_cameras.append(cam)
            
            if missing_cameras:
                errors.append(f"Missing selected cameras: {missing_cameras}. Available cameras must match camera_config.py: {selected_cameras}")
            
            if not available_cameras:
                errors.append(f"No camera images found. Expected: {selected_cameras}")
            
            # If critical errors found, return early
            if errors:
                return False, info, errors
            
            # Get dimensions
            T = f['/action'].shape[0]
            action_dim = f['/action'].shape[1] if len(f['/action'].shape) > 1 else 1
            state_dim = f['/observations/qpos'].shape[1] if len(f['/observations/qpos'].shape) > 1 else 1
            
            # Validate action dataset
            if f['/action'].dtype != np.float32:
                errors.append(f"Action dtype is {f['/action'].dtype}, expected float32")
            
            # Check shape consistency for robot state
            qpos_shape = f['/observations/qpos'].shape
            qvel_shape = f['/observations/qvel'].shape
            
            if qpos_shape[0] != T:
                errors.append(f"qpos length ({qpos_shape[0]}) != action length ({T})")
            
            if qvel_shape[0] != T:
                errors.append(f"qvel length ({qvel_shape[0]}) != action length ({T})")
            
            if f['/observations/qpos'].dtype != np.float32:
                errors.append(f"qpos dtype is {f['/observations/qpos'].dtype}, expected float32")
            
            if f['/observations/qvel'].dtype != np.float32:
                errors.append(f"qvel dtype is {f['/observations/qvel'].dtype}, expected float32")
            
            # Check image data for all available cameras
            for cam in available_cameras:
                img_dataset = f[f'/observations/images/{cam}']
                
                # Determine if images are compressed based on shape
                # Compressed (gzip) images: shape is (T,) or (T, bytes)
                # Raw images: shape is (T, H, W, 3)
                
                # For gzip compressed images in HDF5, the dataset has shape like (T,)
                if len(img_dataset.shape) == 1:
                    if img_dataset.shape[0] != T:
                        errors.append(f"{cam} compressed image count ({img_dataset.shape[0]}) != action length ({T})")
                # For uncompressed raw images, shape should be (T, H, W, 3)
                elif len(img_dataset.shape) == 4:
                    if img_dataset.shape[0] != T:
                        errors.append(f"{cam} image count ({img_dataset.shape[0]}) != action length ({T})")
                    if img_dataset.shape[3] != 3:
                        errors.append(f"{cam} images not RGB: shape[-1]={img_dataset.shape[3]}, expected 3")
                    if img_dataset.dtype != np.uint8:
                        errors.append(f"{cam} image dtype is {img_dataset.dtype}, expected uint8")
                else:
                    errors.append(f"{cam} images have unexpected shape: {img_dataset.shape}")
            
            # Check language instruction
            try:
                if '/language_raw' not in f:
                    errors.append("Missing language_raw dataset")
                else:
                    language_data = f['/language_raw']
                    if len(language_data) == 0:
                        errors.append("Language instruction is empty")
                    else:
                        language_raw = language_data[0]
                        # Language data could be bytes or numpy string
                        if isinstance(language_raw, bytes):
                            language = language_raw.decode('utf-8')
                        elif isinstance(language_raw, np.bytes_):
                            language = language_raw.decode('utf-8')
                        else:
                            language = str(language_raw)
                        
                        if not language or len(language) == 0:
                            errors.append("Language instruction string is empty")
            except Exception as e:
                errors.append(f"Cannot decode language: {e}")
            
            # Check required attributes
            if 'sim' not in f.attrs:
                errors.append("Missing 'sim' attribute (should indicate whether data is from simulator)")
            
            # Collect info
            info = {
                'timesteps': T,
                'action_dim': action_dim,
                'state_dim': state_dim,
                'cameras': available_cameras,
                'language': language if len(available_cameras) > 0 else 'N/A',
                'is_sim': f.attrs.get('sim', 'N/A'),
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
            
            # Get image resolution if available
            first_cam = available_cameras[0]
            img_data = f[f'/observations/images/{first_cam}']
            if len(img_data.shape) == 4:
                info['image_shape'] = f"{img_data.shape[1]}x{img_data.shape[2]}"
            else:
                info['image_shape'] = "compressed (gzip)"
            
            return len(errors) == 0, info, errors
            
    except Exception as e:
        errors.append(f"Failed to open or read file: {e}")
        return False, info, errors


def print_colored(text, color='green'):
    """Print colored text to terminal."""
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['end']}")


def main():
    parser = argparse.ArgumentParser(description='Validate HDF5 dataset files')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing HDF5 files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Print detailed information for each file')
    parser.add_argument('--extension', type=str, default='.hdf5',
                       help='File extension to search for (default: .hdf5)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print_colored(f"Error: Directory does not exist: {data_dir}", 'red')
        sys.exit(1)
    
    # Find all HDF5 files
    hdf5_files = sorted(list(data_dir.glob(f"*{args.extension}")))
    
    if not hdf5_files:
        print_colored(f"Warning: No {args.extension} files found in {data_dir}", 'yellow')
        sys.exit(1)
    
    print(f"Found {len(hdf5_files)} HDF5 files in {data_dir}")
    print("=" * 80)
    print()
    
    valid_count = 0
    invalid_count = 0
    total_timesteps = 0
    total_size_mb = 0
    
    for i, filepath in enumerate(hdf5_files):
        if args.verbose:
            print(f"[{i+1}/{len(hdf5_files)}] Validating: {filepath.name}")
        
        is_valid, info, errors = validate_hdf5_file(filepath, args.verbose)
        
        if is_valid:
            valid_count += 1
            total_timesteps += info.get('timesteps', 0)
            total_size_mb += info.get('file_size_mb', 0)
            
            if args.verbose:
                print_colored(f"  ✓ Valid", 'green')
                print(f"    Timesteps: {info['timesteps']}")
                print(f"    Action dim: {info['action_dim']}, State dim: {info['state_dim']}")
                print(f"    Cameras: {', '.join(info['cameras'])}")
                print(f"    Image shape: {info.get('image_shape', 'N/A')}")
                print(f"    Language: {info['language'][:50]}...")
                print(f"    Is sim: {info['is_sim']}")
                print(f"    File size: {info['file_size_mb']:.2f} MB")
                print()
            else:
                print_colored(f"  ✓ {filepath.name}", 'green')
        else:
            invalid_count += 1
            print_colored(f"  ✗ Invalid: {filepath.name}", 'red')
            for error in errors:
                print_colored(f"    - {error}", 'red')
            print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(hdf5_files)}")
    print_colored(f"Valid files: {valid_count}", 'green' if invalid_count == 0 else 'yellow')
    if invalid_count > 0:
        print_colored(f"Invalid files: {invalid_count}", 'red')
    print()
    print(f"Total timesteps: {total_timesteps}")
    print(f"Total dataset size: {total_size_mb:.2f} MB")
    print(f"Average file size: {total_size_mb/len(hdf5_files):.2f} MB")
    
    if valid_count > 0:
        print()
        print_colored("✓ Dataset is ready for training!", 'green')
        print()
        print("Update scripts/aloha_scripts/constants.py:")
        print(f"  'dataset_dir': ['{data_dir.absolute()}']")
    else:
        print()
        print_colored("✗ No valid files found. Please fix the errors above.", 'red')
        sys.exit(1)


if __name__ == '__main__':
    main()
