import numpy as np
import collections
import os

from diffusion_policy.env.aloha.constants import DT, XML_DIR, START_ARM_POSE
from diffusion_policy.env.aloha.constants import PUPPET_GRIPPER_POSITION_CLOSE
from diffusion_policy.env.aloha.constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from diffusion_policy.env.aloha.constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from diffusion_policy.env.aloha.constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from diffusion_policy.env.aloha.env_utils import sample_box_pose, sample_box_no_rand_pose, sample_insertion_pose, sample_box_pose_large, sample_insertion_pose_large, sample_box_rand_test_pose, sample_box_rand_train_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed


def make_ee_sim_env(task_name, args):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_history_transfer_cube" in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = HistoryTransferCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_pickandplace_samepos_scripted" in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceAtSamePosCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_pickandplace_origin_scripted" in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceAtOriginCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_singlearm_pickandplace_origin_scripted" in task_name:
        xml_path = os.path.join(XML_DIR, f'singlearm_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = SignleArmPickAndPlaceAtOriginCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_singlearm_pickandplace_twomodes_scripted" in task_name:
        xml_path = os.path.join(XML_DIR, f'singlearm_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = SignleArmPickAndPlaceTwoModesCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif "sim_pickandplace_twomodes_scripted" in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceAtTwoModesCubeEETask(random=False, rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_history_pickandplace_transfer_cube_scripted' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = HistoryPickAndPlaceTransferCubeEETask(random=False,rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)   
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False,rand_type=args["rand_type"])
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:16] = START_ARM_POSE

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class SingleArmViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        # a_len = len(action) // 2
        action_right = action
        # action_right = action[a_len:]

        # set mocap position and quat
        # left
        # np.copyto(physics.data.mocap_pos[0], action_left[:3])
        # np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[0], action_right[:3])
        np.copyto(physics.data.mocap_quat[0], action_right[3:7])

        # set gripper
        # g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.named.data.qpos[:8] = START_ARM_POSE[8:]

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        # np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.5, 0.29525084])
        # np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[0], np.array([0.31718881, 0.49999888, 0.29525084]))
        np.copyto(physics.data.mocap_quat[0],  [1, 0, 0, 0])

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        # left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[:8]
        # left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        # left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        # left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[:8]
        # left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        # left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')
        # used in scripted policy to obtain starting pose
        # obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


      
class PickAndPlaceAtOriginCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        self.cube_pose_start = np.array([0.1,0.5, 0.01])
        self.reached_rest = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        curr_box_pos = self.get_observation(physics)["env_state"]

        distance_to_target = np.linalg.norm(curr_box_pos[:3] - self.cube_pose_start)
        arm_rest = False
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and not touch_table and distance_to_target > 0.15: # attempted transfer
            reward = 3
            self.reached_rest = True
        if distance_to_target < 0.05 and touch_table and self.reached_rest: # successful transfer
            reward = 4

        return reward

 
class SignleArmPickAndPlaceTwoModesCubeEETask(SingleArmViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        self.cube_pose_target = get_target_pose(cube_pose[:3])

        self.reached_rest = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[8:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        curr_box_pos = self.get_observation(physics)["env_state"]

        distance_to_target = np.linalg.norm(curr_box_pos[:3] - self.cube_pose_target)
        arm_rest = False
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and not touch_table and distance_to_target > 0.15: # attempted transfer
            reward = 3
            self.reached_rest = True
        if distance_to_target < 0.05 and touch_table and self.reached_rest: # successful transfer
            reward = 4

        return reward

class SignleArmPickAndPlaceAtOriginCubeEETask(SingleArmViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        self.cube_pose_start = np.array([0.1,0.5, 0.01])
        self.reached_rest = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[8:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        curr_box_pos = self.get_observation(physics)["env_state"]

        distance_to_target = np.linalg.norm(curr_box_pos[:3] - self.cube_pose_start)
        arm_rest = False
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and not touch_table and distance_to_target > 0.15: # attempted transfer
            reward = 3
            self.reached_rest = True
        if distance_to_target < 0.05 and touch_table and self.reached_rest: # successful transfer
            reward = 4

        return reward


def get_target_pose(box_pose):
    center1 = np.array([0.1, 0.4, 0.05])
    center2 = np.array([0.1, 0.6, 0.05])

    if np.linalg.norm(center1 - box_pose) < np.linalg.norm(center2 - box_pose):
        return center1
    else:
        return center2

class PickAndPlaceAtTwoModesCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        cube_pose_start = cube_pose[:3]
        self.cube_pose_target = get_target_pose(cube_pose_start)
        self.reached_rest = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        curr_box_pos = self.get_observation(physics)["env_state"]

        distance_to_target = np.linalg.norm(curr_box_pos[:3] - self.cube_pose_target)
        arm_rest = False
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and not touch_table and distance_to_target > 0.15: # attempted transfer
            reward = 3
            self.reached_rest = True
        if distance_to_target < 0.05 and touch_table and self.reached_rest: # successful transfer
            reward = 4

        return reward

class PickAndPlaceAtSamePosCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        self.cube_pose_start = cube_pose[:3]
        self.reached_rest = False
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        curr_box_pos = self.get_observation(physics)["env_state"]

        distance_to_target = np.linalg.norm(curr_box_pos[:3] - self.cube_pose_start)
        arm_rest = False
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_right_gripper and not touch_table and distance_to_target > 0.15: # attempted transfer
            reward = 3
            self.reached_rest = True
        if distance_to_target < 0.05 and touch_table and self.reached_rest: # successful transfer
            reward = 4

        return reward

class HistoryPickAndPlaceTransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 7

        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        self.touched_table = 0
        self.solved_step = False
        self.target_pose = self.get_observation(physics)["qpos"][:8]
        self.current_timestep = 0
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        if self.current_timestep == 0:
            self.current_timestep += 1
            self.target_pose = self.get_observation(physics)["qpos"][:7]

        distance_to_target = np.linalg.norm(self.get_observation(physics)["qpos"][:7] - self.target_pose)
        gripper_close_to_target = distance_to_target < 0.8

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
            self.t += 1
        if touch_right_gripper and self.t > 100 and not touch_table:
            reward = 5
        if self.t > 100 and touch_table:
            self.touched_table += 1
            reward = 6
        if self.touched_table > 50 and touch_table and gripper_close_to_target:
            reward = 7
            self.solved_step = True
        if touch_right_gripper and self.solved_step and not touch_left_gripper and not touch_table:
            reward = 8

        return reward

class HistoryTransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 5


        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")
        self.t = 0
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
            self.t += 1
        if touch_right_gripper and self.t > 200 and not touch_table:
            reward = 5

        return reward


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4


        if rand_type == "large_rand":
            self.rand_pose = sample_box_pose_large # used in sim reset
        elif rand_type == "rand_train":
            self.rand_pose =  sample_box_rand_train_pose # used in sim reset
        elif rand_type == "rand_test":
            self.rand_pose =  sample_box_rand_test_pose # used in sim reset
        elif rand_type == "no_rand":
            self.rand_pose =  sample_box_no_rand_pose # used in sim reset
        elif rand_type == "low_rand":    
            self.rand_pose = sample_box_pose # used in sim reset
        else:
            raise NotImplementedError

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = self.rand_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None, rand_type=False):
        super().__init__(random=random)
        self.max_reward = 4
        if rand_type == "large_rand":
            self.rand_pose = sample_insertion_pose_large
        elif rand_type == "low_rand":
            self.rand_pose = sample_insertion_pose
        else:
            raise NotImplementedError
        

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = self.rand_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward