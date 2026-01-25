
##################### Setting of training data #####################################

# DATA_DIR = '/path/to/your/data_dir'

# Import selected cameras from camera_config
import sys
import os
scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, scripts_dir)
from camera_config import get_camera_names

TASK_CONFIGS = {
    'example_task_config': {
        'dataset_dir': [
            "/data/my_task",  # TODO: Update this path to your HDF5 files
        ],
        'episode_len': 500,
        # Uses camera names from camera_config
        'camera_names': get_camera_names(),  
        'stats_dir': None,  # Optional: use different dir for normalization stats
        'sample_weights': None,  # Optional: weights for sampling different datasets
        'train_ratio': 0.95,  # 95% train, 5% validation
        'name_filter': lambda n: True  # Filter function for dataset files
    },
    'vla_diff_head_lora': {
        'dataset_dir': [
            "/home/hemanthm/Desktop/GRASP/GRASP/data/metaworld_pick-place-v3_front",  # Dummy dataset for testing
        ],
        'episode_len': 500,
        # Uses camera names from camera_config
        'camera_names': get_camera_names(),
        'stats_dir': None,  # Optional: use different dir for normalization stats
        'sample_weights': None,  # Optional: weights for sampling different datasets
        'train_ratio': 0.95,  # 95% train, 5% validation
        'name_filter': lambda n: n.endswith('.hdf5')  # Only load .hdf5 files
    },
    'metaworld_task': {
        'dataset_dir': [
            "/home/hemanthm/Desktop/GRASP/GRASP/data/metaworld_pick-place-v3_front",
        ],
        'episode_len': 500, # Matched to generation script
        # Uses camera names from camera_config - will include only selected cameras
        'camera_names': get_camera_names(),
        'stats_dir': None,
        'sample_weights': None,
        'train_ratio': 0.95,
        'name_filter': lambda n: n.endswith('.hdf5')
    },
    'metaworld_hf': {
        'hf_dataset_name': 'aadarshram/metaworld-pick-place-v3',  # HuggingFace dataset
        'episode_len': 500,  # Max episode length
        'camera_names': get_camera_names(),  # Uses camera_config (should be ['right'] for corner2)
        'stats_dir': None,
        'sample_weights': None,
        'train_ratio': 0.95,  # 95% train, 5% validation
        'use_hf_dataset': True,  # Flag to indicate HuggingFace dataset
    },
}
####################################################################################

#!!!!!!!!!!!!!!!!!!!!!!Followings are copied from aloha which are not used!!!!!!!!!!!!!!!!!!!!!!
### ALOHA fixed constants
DT = 0.02

FPS = 50

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2