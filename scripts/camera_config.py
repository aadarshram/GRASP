"""
Global Camera Configuration for GRASP

This file allows you to select which camera views to use throughout the codebase.
The selection is automatically propagated to:
- Data generation (generate_metaworld_data.py)
- Dataset validation (validate_dataset.py)
- Dataset visualization (visualize_dataset.py)
- Model evaluation (eval.py, eval_metaworld.py)
- Training data loading (datasets.py via constants.py)

Camera Definitions for MetaWorld:
- 'top': Top-down view of the workspace (Camera ID 0)
- 'left': Left side view (Camera ID 1)
- 'right': Right side view (Camera ID 2)
- 'front': Front-facing view with custom position (Camera ID 4)
- 'gripper': Wrist/end-effector camera (Camera ID 5)

Example Usage:
    from scripts.camera_config import SELECTED_CAMERAS, get_camera_names
    
    cameras = get_camera_names()  # Returns ['top', 'left', 'right']
    for cam in cameras:
        print(f"Using camera: {cam}")
"""

# ============================================================================
# CAMERA SELECTION - MODIFY THIS TO SELECT CAMERAS
# ============================================================================

# All available cameras in MetaWorld environment
METAWORLD_CAMERAS = {
    'top': {'id': 0, 'name': 'topview', 'description': 'Top-down view of workspace'},
    'left': {'id': 1, 'name': 'corner', 'description': 'Left side view'},
    'right': {'id': 2, 'name': 'corner2', 'description': 'Right side view'},
    'front': {'id': 4, 'name': 'front_custom', 'description': 'Front-facing view'},
    'gripper': {'id': 5, 'name': 'behindGripper', 'description': 'Wrist/end-effector camera'},
}

# ============================================================================
# SELECT YOUR CAMERAS HERE
# ============================================================================
# Choose which cameras to use. Only selected cameras will be:
# - Rendered during data generation
# - Saved in HDF5 files
# - Loaded during training
# - Used during evaluation
# - Displayed in visualizations

# Option 1: All 5 cameras (comprehensive data)
# SELECTED_CAMERAS = ['top', 'left', 'right', 'front', 'gripper']

# Option 2: Standard 3 cameras (left, right, top) - Most common
# SELECTED_CAMERAS = ['top', 'left', 'right']

# Option 3: Only front camera (minimal)
# SELECTED_CAMERAS = ['front']

# Option 4: Left and right only (multi-view without top)
# SELECTED_CAMERAS = ['left', 'right']

# Option 5: Include gripper view for manipulation tasks
# SELECTED_CAMERAS = ['top', 'left', 'right', 'gripper']

# Option 6: Right camera only (corner2) - for HuggingFace metaworld-pick-place-v3 dataset
SELECTED_CAMERAS = ['right']  # 'right' corresponds to 'corner2' in MetaWorld

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_camera_names() -> list[str]:
    """
    Returns the list of selected camera names.
    
    Returns:
        list: Selected camera names (e.g., ['top', 'left', 'right'])
    """
    return SELECTED_CAMERAS.copy()


def get_camera_ids():
    """
    Returns a mapping of camera names to their MetaWorld IDs.
    
    Returns:
        dict: Mapping {camera_name: camera_id} for selected cameras
    """
    return {cam: METAWORLD_CAMERAS[cam]['id'] for cam in SELECTED_CAMERAS}


def get_camera_mujoco_names():
    """
    Returns a mapping of camera names to their MuJoCo names.
    Used for rendering from specific cameras.
    
    Returns:
        dict: Mapping {camera_name: mujoco_name} for selected cameras
    """
    return {cam: METAWORLD_CAMERAS[cam]['name'] for cam in SELECTED_CAMERAS}


def validate_camera_selection(cameras):
    """
    Validates that all selected cameras are available.
    
    Args:
        cameras (list): List of camera names to validate
        
    Returns:
        bool: True if all cameras are valid, False otherwise
    """
    for cam in cameras:
        if cam not in METAWORLD_CAMERAS:
            print(f"ERROR: Camera '{cam}' not found in METAWORLD_CAMERAS")
            return False
    return True


def get_camera_info(camera_name):
    """
    Gets detailed information about a specific camera.
    
    Args:
        camera_name (str): Name of the camera
        
    Returns:
        dict or None: Camera information if found, None otherwise
    """
    return METAWORLD_CAMERAS.get(camera_name)


def num_cameras():
    """
    Returns the number of selected cameras.
    
    Returns:
        int: Number of selected cameras
    """
    return len(SELECTED_CAMERAS)


# Validate camera selection on import
if not validate_camera_selection(SELECTED_CAMERAS):
    raise ValueError(f"Invalid camera selection in SELECTED_CAMERAS: {SELECTED_CAMERAS}")

if __name__ == "__main__":
    print("=" * 70)
    print("GRASP Camera Configuration")
    print("=" * 70)
    print(f"\nSelected Cameras: {get_camera_names()}")
    print(f"Number of Cameras: {num_cameras()}")
    print(f"\nCamera IDs: {get_camera_ids()}")
    print(f"\nMuJoCo Names: {get_camera_mujoco_names()}")
    print("\nCamera Details:")
    for cam in get_camera_names():
        info = get_camera_info(cam)
        print(f"  - {cam} (ID: {info['id']}): {info['description']}")
    print("\n" + "=" * 70)
