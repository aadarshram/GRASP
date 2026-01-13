import h5py
import cv2
import argparse
import numpy as np
import time
import os
import sys

# Add scripts directory to path for camera_config import
scripts_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, scripts_dir)
from camera_config import get_camera_names, METAWORLD_CAMERAS


def print_hdf5_structure(file_path):
    """
    Print the complete structure and metadata of an HDF5 file.
    """
    print("\n" + "="*80)
    print(f"HDF5 File Structure: {file_path}")
    print("="*80)
    
    with h5py.File(file_path, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                compression = obj.compression if hasattr(obj, 'compression') else None
                comp_str = f" [compressed: {compression}]" if compression else ""
                print(f"{indent}├── {name} {shape} dtype={dtype}{comp_str}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}├── {name}/ (Group)")
        
        print("\nDatasets and Groups:")
        f.visititems(print_structure)
        
        print("\nRoot Attributes:")
        for attr_name, attr_value in f.attrs.items():
            print(f"  └── {attr_name}: {attr_value} (type: {type(attr_value).__name__})")
    
    print("="*80 + "\n")


def unpack_hdf5_file(file_path, verbose=True):
    """
    Unpack and extract all information from an HDF5 file.
    
    Returns:
        dict: Contains all unpacked data and metadata
    """
    with h5py.File(file_path, 'r') as f:
        data = {}
        
        # Extract metadata (root attributes)
        data['metadata'] = {
            'is_sim': f.attrs.get('sim', False),
            'is_compressed': f.attrs.get('compress', False),
            'file_path': file_path,
            'file_size_mb': os.path.getsize(file_path) / (1024**2)
        }
        
        # Extract action data
        if '/action' in f:
            data['action'] = f['/action'][:]
            data['action_shape'] = f['/action'].shape
            data['action_dtype'] = str(f['/action'].dtype)
            data['action_stats'] = {
                'min': np.min(data['action']),
                'max': np.max(data['action']),
                'mean': np.mean(data['action']),
                'std': np.std(data['action'])
            }
        
        # Extract observations
        data['observations'] = {}
        
        if '/observations/qpos' in f:
            data['observations']['qpos'] = f['/observations/qpos'][:]
            data['observations']['qpos_shape'] = f['/observations/qpos'].shape
            data['observations']['qpos_dtype'] = str(f['/observations/qpos'].dtype)
            data['observations']['qpos_stats'] = {
                'min': np.min(data['observations']['qpos']),
                'max': np.max(data['observations']['qpos']),
                'mean': np.mean(data['observations']['qpos']),
                'std': np.std(data['observations']['qpos'])
            }
        
        if '/observations/qvel' in f:
            data['observations']['qvel'] = f['/observations/qvel'][:]
            data['observations']['qvel_shape'] = f['/observations/qvel'].shape
            data['observations']['qvel_dtype'] = str(f['/observations/qvel'].dtype)
            data['observations']['qvel_stats'] = {
                'min': np.min(data['observations']['qvel']),
                'max': np.max(data['observations']['qvel']),
                'mean': np.mean(data['observations']['qvel']),
                'std': np.std(data['observations']['qvel'])
            }
        
        # Extract images
        data['images'] = {}
        if '/observations/images' in f:
            img_group = f['/observations/images']
            for cam_name in img_group.keys():
                cam_data = img_group[cam_name]
                data['images'][cam_name] = cam_data[:]
                data['images'][f'{cam_name}_shape'] = cam_data.shape
                data['images'][f'{cam_name}_dtype'] = str(cam_data.dtype)
                data['images'][f'{cam_name}_compression'] = cam_data.compression
                data['images'][f'{cam_name}_size_mb'] = (np.prod(cam_data.shape) * np.dtype(cam_data.dtype).itemsize) / (1024**2)
        
        # Extract language instruction
        data['language'] = None
        if '/language_raw' in f:
            language_data = f['/language_raw'][0]
            if isinstance(language_data, bytes):
                data['language'] = language_data.decode('utf-8')
            else:
                data['language'] = str(language_data)
        
        # Episode statistics
        if data['action'] is not None:
            data['episode_length'] = len(data['action'])
        elif data['observations'].get('qpos') is not None:
            data['episode_length'] = len(data['observations']['qpos'])
        
        if verbose:
            print_unpacked_info(data)
        
        return data


def print_unpacked_info(data):
    """
    Print all unpacked information from HDF5 file.
    """
    print("\n" + "="*80)
    print("UNPACKED HDF5 FILE INFORMATION")
    print("="*80)
    
    # Metadata
    print("\n[METADATA]")
    meta = data.get('metadata', {})
    print(f"  File Path: {meta.get('file_path', 'N/A')}")
    print(f"  File Size: {meta.get('file_size_mb', 0):.2f} MB")
    print(f"  From Simulation: {meta.get('is_sim', False)}")
    print(f"  Compressed: {meta.get('is_compressed', False)}")
    print(f"  Episode Length: {data.get('episode_length', 'N/A')} timesteps")
    
    # Language
    print("\n[LANGUAGE INSTRUCTION]")
    print(f"  Task: {data.get('language', 'N/A')}")
    
    # Actions
    print("\n[ACTIONS]")
    if 'action_shape' in data:
        print(f"  Shape: {data['action_shape']}")
        print(f"  Data Type: {data['action_dtype']}")
        print(f"  Dimensions: timesteps={data['action_shape'][0]}, action_dim={data['action_shape'][1]}")
        stats = data.get('action_stats', {})
        print(f"  Statistics:")
        print(f"    Min: {stats.get('min', 'N/A'):.4f}")
        print(f"    Max: {stats.get('max', 'N/A'):.4f}")
        print(f"    Mean: {stats.get('mean', 'N/A'):.4f}")
        print(f"    Std: {stats.get('std', 'N/A'):.4f}")
    
    # Observations
    obs = data.get('observations', {})
    
    print("\n[ROBOT STATE - JOINT POSITIONS (qpos)]")
    if 'qpos_shape' in obs:
        print(f"  Shape: {obs['qpos_shape']}")
        print(f"  Data Type: {obs['qpos_dtype']}")
        print(f"  Dimensions: timesteps={obs['qpos_shape'][0]}, dof={obs['qpos_shape'][1]}")
        stats = obs.get('qpos_stats', {})
        print(f"  Statistics:")
        print(f"    Min: {stats.get('min', 'N/A'):.4f}")
        print(f"    Max: {stats.get('max', 'N/A'):.4f}")
        print(f"    Mean: {stats.get('mean', 'N/A'):.4f}")
        print(f"    Std: {stats.get('std', 'N/A'):.4f}")
    
    print("\n[ROBOT STATE - JOINT VELOCITIES (qvel)]")
    if 'qvel_shape' in obs:
        print(f"  Shape: {obs['qvel_shape']}")
        print(f"  Data Type: {obs['qvel_dtype']}")
        print(f"  Dimensions: timesteps={obs['qvel_shape'][0]}, dof={obs['qvel_shape'][1]}")
        stats = obs.get('qvel_stats', {})
        print(f"  Statistics:")
        print(f"    Min: {stats.get('min', 'N/A'):.4f}")
        print(f"    Max: {stats.get('max', 'N/A'):.4f}")
        print(f"    Mean: {stats.get('mean', 'N/A'):.4f}")
        print(f"    Std: {stats.get('std', 'N/A'):.4f}")
    
    # Images
    images = data.get('images', {})
    print("\n[CAMERA IMAGES]")
    camera_names = [k for k in images.keys() if k.endswith('_shape')]
    for cam_key in sorted(camera_names):
        cam_name = cam_key.replace('_shape', '')
        shape = images.get(cam_key, 'N/A')
        dtype = images.get(f'{cam_name}_dtype', 'N/A')
        compression = images.get(f'{cam_name}_compression', None)
        size_mb = images.get(f'{cam_name}_size_mb', 0)
        
        comp_str = f" [Compression: {compression}]" if compression else ""
        print(f"\n  Camera: {cam_name.upper()}")
        print(f"    Shape: {shape}")
        print(f"    Data Type: {dtype}")
        print(f"    Dimensions: timesteps={shape[0]}, height={shape[1]}, width={shape[2]}, channels={shape[3]}")
        print(f"    Uncompressed Size: {size_mb:.2f} MB{comp_str}")
    
    print("\n" + "="*80 + "\n")


def visualize_episode(file_path, show_info=True):
    """Visualize episode with optional information display."""
    
    # Unpack and display all file information
    if show_info:
        unpack_hdf5_file(file_path, verbose=True)
    
    print(f"Visualizing {file_path}")
    with h5py.File(file_path, 'r') as f:
        # Get selected cameras from camera_config
        selected_cameras = get_camera_names()
        
        # Check which selected cameras are available in this file
        cameras = []
        for cam in selected_cameras:
            if f'/observations/images/{cam}' in f:
                cameras.append(cam)
        
        if not cameras:
            # Fallback: check any available cameras
            all_possible = list(METAWORLD_CAMERAS.keys())
            for cam in all_possible:
                if f'/observations/images/{cam}' in f:
                    cameras.append(cam)
            if cameras:
                print(f"Warning: Selected cameras {selected_cameras} not found. Using available: {cameras}")
            else:
                print("No images found in file.")
                return

        print(f"Visualizing cameras: {cameras}")
        
        # Get language instruction
        language = None
        if '/language_raw' in f:
            language_data = f['/language_raw'][0]
            if isinstance(language_data, bytes):
                language = language_data.decode('utf-8')
            else:
                language = str(language_data)
        
        # Get episode length from first available camera
        T = f[f'/observations/images/{cameras[0]}'].shape[0]
        
        print(f"Episode length: {T}")
        print("Press 'q' to quit, any other key to speed up.")
        
        for t in range(T):
            imgs = []
            frame_info = f"Frame: {t+1}/{T}"
            
            # Get robot state at this timestep if available
            state_str = ""
            if '/observations/qpos' in f:
                qpos = f['/observations/qpos'][t]
                qpos_str = np.array2string(qpos, precision=3, separator=',')[:70]
                state_str += f"  qpos: {qpos_str}..."
            if '/observations/qvel' in f:
                qvel = f['/observations/qvel'][t]
                if state_str:
                    state_str += "\n"
                qvel_str = np.array2string(qvel, precision=3, separator=',')[:70]
                state_str += f"  qvel: {qvel_str}..."
            
            # Get action at this timestep if available
            action_str = ""
            if '/action' in f:
                action = f['/action'][t]
                action_str_full = np.array2string(action, precision=3, separator=',')[:70]
                action_str = f"  action: {action_str_full}..."
            
            for cam in cameras:
                cam_data = f[f'/observations/images/{cam}'][t]
                
                # Handle compressed images (bytes) vs raw arrays
                if isinstance(cam_data, bytes) or (isinstance(cam_data, np.ndarray) and cam_data.dtype == np.uint8 and len(cam_data.shape) == 1):
                    # Image is compressed (gzip from HDF5)
                    if isinstance(cam_data, np.ndarray):
                        cam_data = cam_data.tobytes()
                    img = cv2.imdecode(np.frombuffer(cam_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    # Image is raw array
                    img = cam_data
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        # Check if RGB (needs conversion to BGR for OpenCV)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                if img is None:
                    print(f"Warning: Could not decode image for camera '{cam}' at frame {t}")
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Create label with camera ID from config
                cam_info = METAWORLD_CAMERAS.get(cam, {})
                cam_id = cam_info.get('id', '?')
                label = f"{cam.capitalize()} (ID:{cam_id})"
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                imgs.append(img)
            
            display_img = np.hstack(imgs)
            
            # Add text overlay with frame info and state data
            h, w = display_img.shape[:2]
            # Create space for text
            text_height = 150 if (state_str or action_str) else 50
            display_with_text = np.vstack([
                np.ones((text_height, w, 3), dtype=np.uint8) * 30,
                display_img
            ])
            
            # Add text
            y_offset = 20
            cv2.putText(display_with_text, frame_info, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 25
            
            if language:
                cv2.putText(display_with_text, f"Task: {language[:60]}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                y_offset += 25
            
            if state_str:
                for line in state_str.split('\n'):
                    cv2.putText(display_with_text, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    y_offset += 20
            
            if action_str:
                cv2.putText(display_with_text, action_str, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
            
            cv2.imshow('Episode Visualization (with State & Action Info)', display_with_text)
            
            key = cv2.waitKey(50)  # 50ms per frame
            if key == ord('q'):
                return

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Visualize HDF5 episodes with complete information unpacking')
    parser.add_argument('--file', type=str, help='Path to specific HDF5 file to visualize')
    parser.add_argument('--dir', type=str, default='data/metaworld_dataset', help='Directory to search for HDF5 files')
    parser.add_argument('--info-only', action='store_true', help='Only print information, do not visualize videos')
    parser.add_argument('--no-info', action='store_true', help='Skip printing information, go straight to visualization')
    args = parser.parse_args()
    
    if args.file:
        show_info = not args.no_info
        visualize_episode(args.file, show_info=show_info)
    else:
        if not os.path.exists(args.dir):
            print(f"Error: Directory does not exist: {args.dir}")
            return
        
        files = sorted([f for f in os.listdir(args.dir) if f.endswith('.hdf5')])
        if not files:
            print(f"No HDF5 files found in {args.dir}")
            return
        
        print(f"Found {len(files)} HDF5 files. Starting visualization...\n")
        for f in files:
            path = os.path.join(args.dir, f)
            try:
                show_info = not args.no_info
                visualize_episode(path, show_info=show_info)
                if args.info_only:
                    continue
            except Exception as e:
                print(f"Error visualizing {f}: {e}")
            print("Next episode...\n")


if __name__ == "__main__":
    main()

