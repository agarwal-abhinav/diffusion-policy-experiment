from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
import copy
from diffusion_policy.env.aloha.env_utils import sample_box_pose, sample_box_no_rand_pose, sample_insertion_pose, sample_box_pose_large, sample_insertion_pose_large, sample_box_rand_test_pose, sample_box_rand_train_pose

class AlohaImageWrapper(gym.Env):
    def __init__(self, 
        env,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='top',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.max_episode_steps = 500 # TODO try to get it properly
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('top'):
                min_value, max_value = 0, 1
            elif key.endswith('wrist'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -3, 3
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        obs_dict = copy.deepcopy(raw_obs.observation)

        for img_key in obs_dict["images"].keys():
            new_image = (obs_dict["images"][img_key] / 255).transpose((2,0,1))
            obs_dict[img_key] = new_image

        del obs_dict["images"]

        self.render_cache = obs_dict[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = obs_dict[key]

        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self, seed=None, options=None):
        self.t = 0
        BOX_POSE = sample_box_rand_train_pose()
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                # raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
                raw_obs = self.env.reset()
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = raw_obs.observation["env_state"]
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        self.t+=1
        try:
            ts = self.env.step(action)
        except:
            print("Environment failed!!")
            raw_obs = self.env.reset()
            obs = self.get_observation(raw_obs)
            return obs, 0, True, {}
        obs = self.get_observation(ts)
        return obs, ts.reward, False, {}
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dptest/transformer_aloha.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']
    task_name = "single_arm_origin_"

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dptest/data/single_arm_origin_train_rand_with_images_wrist/demos.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)
    from sim_env import make_sim_env
    aloha_env = make_sim_env(
            task_name=task_name, 
        )

    wrapper = AlohaImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)
    plt.savefig("~/dptest/imagewrapper.png")

    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])

if __name__ == "__main__":
    test()