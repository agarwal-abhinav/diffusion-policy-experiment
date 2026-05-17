import pathlib

### Task parameters
DATA_DIR = 'data'
MIMICGEN_DATA_DIR = 'data'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'state_dim': 14,
        'env_state_dim': 7,
        'action_dim': 14,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_history_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_history_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 800,
        'state_dim': 14,
        'action_dim': 14,
        'env_state_dim': 7,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_history_pickandplace_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_history_pickandplace_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 950,
        'env_state_dim': 7,
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_cube_scripted_hist':{
        'dataset_dir': DATA_DIR + '/sim_cube_scripted_hist',
        'num_episodes': 10,
        'episode_len': 500,
        'state_dim': 14,
        'action_dim': 14,
        'env_state_dim': 7,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_pickandplace_samepos_scripted':{
        'dataset_dir': DATA_DIR + '/sim_pickandplace_samepos_scripted',
        'num_episodes': 50,
        'episode_len': 600,
        'state_dim': 14,
        'action_dim': 14,
        'env_state_dim': 7,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_pickandplace_origin_scripted':{
        'dataset_dir': DATA_DIR + '/sim_pickandplace_samepos_scripted',
        'num_episodes': 50,
        'episode_len': 500,
        'state_dim': 14,
        'action_dim': 14,
        'env_state_dim': 7,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_singlearm_pickandplace_twomodes_scripted':{
        'dataset_dir': DATA_DIR + '/sim_pickandplace_twomodes_scripted',
        'num_episodes': 50,
        'episode_len': 500,
        'state_dim': 7,
        'action_dim': 7,
        'env_state_dim': 7,
        'camera_names': ['top', "right_wrist"]
    },
    'sim_singlearm_pickandplace_origin_scripted':{
        'dataset_dir': DATA_DIR + '/sim_pickandplace_samepos_scripted',
        'num_episodes': 50,
        'episode_len': 500,
        'state_dim': 7,
        'action_dim': 7,
        'env_state_dim': 7,
        'camera_names': ['top', "right_wrist"]
    },
    'sim_pickandplace_twomodes_scripted':{
        'dataset_dir': DATA_DIR + '/sim_pickandplace_samepos_scripted',
        'num_episodes': 50,
        'episode_len': 500,
        'state_dim': 14,
        'env_state_dim': 7,
        'action_dim': 14,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'state_dim': 14,
        'action_dim': 14,
        'env_state_dim': 7,
        'camera_names': ['top']
    },
    'mimicgen_square_0':{
        'dataset_dir': '',
        'num_episodes': 300,
        'episode_len': 1500,
        'state_dim': 9,
        'action_dim': 7,
        'env_state_dim': 14,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_square_1':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 200,
        'state_dim': 9,
        'action_dim': 7,
        'env_state_dim': 14,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_square_2':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 200,
        'state_dim': 9,
        'env_state_dim': 14,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_hammer_cleanup_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 350,
        'state_dim': 9,
        'env_state_dim': 14,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_kitchen_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 700,
        'state_dim': 9,
        'env_state_dim': 14,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_lift_0':{
        'dataset_dir': '',
        'num_episodes': 300,
        'episode_len': 2000,
        'state_dim': 9,
        'env_state_dim': 14,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_can_0':{
        'dataset_dir': '',
        'num_episodes': 300,
        'episode_len': 2000,
        'state_dim': 9,
        'env_state_dim': 14,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_coffee_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 2000,
        'state_dim': 9,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_transport_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 2000,
        'state_dim': 9,
        'action_dim': 14,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_coffee_preparation_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 800,
        'state_dim': 9,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_mug_cleanup_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 400,
        'state_dim': 9,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_nut_assembly_0':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 600,
        'state_dim': 9,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'mimicgen_nut_assembly_1':{
        'dataset_dir': '',
        'num_episodes': 1000,
        'episode_len': 600,
        'state_dim': 9,
        'action_dim': 7,
        'camera_names': ['agentview_image','robot0_eye_in_hand_image']
    },
    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
    'sim_insertion_transport': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 1000,
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'state_dim': 14,
        'action_dim': 14,
        'camera_names': ['top']
    },

    'sim_calvin_debug': {
        'dataset_dir': 'iris/u/marcelto/calvin/dataset/calvin_debug_dataset',
        'num_episodes': 0, # TODO PLACEHOLDER
        'episode_len': 0, # TODO PLACEHOLDER
        'state_dim': 399,
        'action_dim': 7,
        'camera_names': ["cam_1", "cam_2"]
    },
    'sim_calvin_D': {
        'dataset_dir': 'iris/u/marcelto/calvin/dataset/task_D_D',
        'num_episodes': 0, # TODO PLACEHOLDER
        'episode_len': 0, # TODO PLACEHOLDER
        'state_dim': 399,
        'action_dim': 7,
        'camera_names': ["cam_1", "cam_2"]
    },
}

### Simulation envs fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

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