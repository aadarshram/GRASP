import h5py
import cv2
import argparse
import numpy as np
import time

def visualize_episode(file_path):
    print(f"Visualizing {file_path}")
    with h5py.File(file_path, 'r') as f:
        # Check available cameras
        cameras = []
        if '/observations/images/left' in f: cameras.append('left')
        if '/observations/images/right' in f: cameras.append('right')
        if '/observations/images/top' in f: cameras.append('top')
        if '/observations/images/front' in f: cameras.append('front')
        
        if not cameras:
            print("No images found in file.")
            return

        # Get length
        T = f['/observations/images/' + cameras[0]].shape[0]
        
        print(f"Episode length: {T}")
        print("Press 'q' to quit, any other key to speed up.")
        
        for t in range(T):
            imgs = []
            for cam in cameras:
                img = f['/observations/images/' + cam][t]
                if len(img.shape) == 1:
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                # label mapping
                labels = {'left': 'Corner', 'right': 'Corner 2', 'top': 'Top', 'front': 'Front'}
                label = labels.get(cam, cam.capitalize())
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                imgs.append(img)
            
            display_img = np.hstack(imgs)
            cv2.imshow('Episode Visualization', display_img)
            
            key = cv2.waitKey(50) # 50ms per frame
            if key == ord('q'):
                return

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to HDF5 file')
    parser.add_argument('--dir', type=str, default='data/metaworld_dataset', help='Directory to search for files')
    args = parser.parse_args()
    
    if args.file:
        visualize_episode(args.file)
    else:
        import os
        files = sorted([f for f in os.listdir(args.dir) if f.endswith('.hdf5')])
        if not files:
            print(f"No HDF5 files found in {args.dir}")
            return
        
        for f in files:
            path = os.path.join(args.dir, f)
            visualize_episode(path)
            print("Next episode...")

if __name__ == "__main__":
    main()
