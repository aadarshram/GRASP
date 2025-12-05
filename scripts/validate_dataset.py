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


def validate_hdf5_file(filepath, verbose=False):
    """
    Validate that an HDF5 file has the required structure for VLA training.
    
    Args:
        filepath: Path to HDF5 file
        verbose: Print detailed information
    
    Returns:
        (is_valid, info_dict, errors_list)
    """
    required_keys = ['/action', '/observations/qpos', '/observations/qvel', '/language_raw']
    camera_keys = ['/observations/images/left', '/observations/images/right', '/observations/images/top']
    
    errors = []
    info = {}
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required datasets
            for key in required_keys:
                if key not in f:
                    errors.append(f"Missing required dataset: {key}")
            
            # Check cameras
            for key in camera_keys:
                if key not in f:
                    errors.append(f"Missing camera: {key}")
            
            if errors:
                return False, info, errors
            
            # Get dimensions
            T = f['/action'].shape[0]
            action_dim = f['/action'].shape[1]
            state_dim = f['/observations/qpos'].shape[1]
            
            # Check shape consistency
            if f['/observations/qpos'].shape[0] != T:
                errors.append(f"qpos length ({f['/observations/qpos'].shape[0]}) != action length ({T})")
            
            if f['/observations/qvel'].shape[0] != T:
                errors.append(f"qvel length ({f['/observations/qvel'].shape[0]}) != action length ({T})")
            
            # Check image shapes
            is_compressed = f.attrs.get('compress', False)
            for cam in ['left', 'right', 'top']:
                img_data = f[f'/observations/images/{cam}']
                if not is_compressed:
                    if len(img_data.shape) != 4:
                        errors.append(f"{cam} images have wrong shape: {img_data.shape}, expected (T, H, W, 3)")
                    elif img_data.shape[0] != T:
                        errors.append(f"{cam} image count ({img_data.shape[0]}) != action length ({T})")
                    elif img_data.shape[3] != 3:
                        errors.append(f"{cam} images not RGB: shape[-1]={img_data.shape[3]}, expected 3")
                else:
                    # Compressed images
                    if len(img_data) != T:
                        errors.append(f"{cam} compressed image count ({len(img_data)}) != action length ({T})")
            
            # Check language
            try:
                language = f['/language_raw'][0].decode('utf-8')
                if not language or len(language) == 0:
                    errors.append("Language instruction is empty")
            except Exception as e:
                errors.append(f"Cannot decode language: {e}")
            
            # Check attributes
            if 'sim' not in f.attrs:
                errors.append("Missing 'sim' attribute")
            
            # Collect info
            info = {
                'timesteps': T,
                'action_dim': action_dim,
                'state_dim': state_dim,
                'language': f['/language_raw'][0].decode('utf-8') if '/language_raw' in f else 'N/A',
                'is_sim': f.attrs.get('sim', 'N/A'),
                'compressed': is_compressed,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
            
            if not is_compressed and '/observations/images/left' in f:
                img_shape = f['/observations/images/left'].shape
                info['image_shape'] = f"{img_shape[1]}x{img_shape[2]}"
            
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
                print(f"    Image shape: {info.get('image_shape', 'N/A')}")
                print(f"    Language: {info['language'][:50]}...")
                print(f"    Compressed: {info['compressed']}")
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
