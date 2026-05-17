"""
MultimodalSquareWrapper: a 4-stage long-horizon reward wrapper for the
robomimic NutAssembly (SquareNut) environment.

Adapted verbatim from the LDP repo
(https://github.com/long-context-dp/ldp,
 file: diffusion_policy/env_runner/robomimic_longhist_image_runner.py
 lines ~45-129)

The wrapper customises the env in two ways:

1. `reset()` forces a specific initial joint state with the nut placed at a
   uniformly-random (x, y) position in the box x ∈ [-0.2, 0], y ∈ [-0.2, 0.2].
   It also records the "target peg" (the peg closer to the nut), though the
   reward logic below intentionally uses `peg_id=0` regardless.

2. `step()` replaces the standard binary nut-placed reward with a 4-stage
   multi-step reward:
       stage 0 -> 1: nut touches peg 0
       stage 1 -> 2: end-effector reaches "saddle high" position (~ 0.2 m
                     above scene center)
       stage 2 -> 3: nut back on peg 0
       stage 3 -> 4: end-effector reaches "saddle drop" position (at scene
                     center, z = 0.89)
   `reward = 1` only when all 4 stages have been completed; otherwise 0.

The wrapper expects to wrap a `RobomimicImageWrapper`, which itself wraps a
robomimic `EnvRobosuite` over a robosuite `NutAssemblySquare`. It accesses
the underlying robosuite env via `self.env.env.env` (one level for the
RobomimicImageWrapper, one for the robomimic EnvRobosuite, one for the
robosuite env itself) for fields such as `sim.data.body_xpos`,
`obj_body_id['SquareNut']`, `peg1_body_id`, `peg2_body_id`, and `on_peg(...)`.

These attributes are present in robosuite >= 1.2.0 NutAssembly envs
(verified in the `robodiff` conda env at robosuite 1.2.0).

Notes on use:
- Place this wrapper INSIDE `VideoRecordingWrapper` but OUTSIDE
  `RobomimicImageWrapper` in the env stack. See
  `diffusion_policy/env_runner/robomimic_longhist_image_runner.py` for the
  standard wiring.
- This wrapper imports nothing PTP-specific. Test-time consistency selection
  is NOT part of the env definition; if you ever want it, add it at the
  env_runner level as an inference flag.
"""
import numpy as np

from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
)


class MultimodalSquareWrapper(RobomimicImageWrapper):
    def __init__(
        self,
        env: RobomimicImageWrapper,
    ):
        self.env = env
        # self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_observation(self):
        return self.env.get_observation()

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        # state = self.env.get_state()['states']

        nut_pos = np.random.uniform([-0.2, -0.2], [0, 0.2], size=(2))

        # Hardcoded 45-element mujoco state for NutAssemblySquare.
        # Indices [10:12] are overwritten with the random nut (x, y) below.
        reset_state = np.array(
            [
                 0.        , -0.02921895,  0.17810908,  0.02728627, -2.63967499,
                -0.01431297,  2.9502351 ,  0.77126893,  0.020833  , -0.020833  ,
                -0.11083517,  0.11445349,  0.89      , -0.98421243,  0.        ,
                 0.        ,  0.17699121, 10.        , 10.        , 10.        ,
                 1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            ]
        )

        self.rew = 0
        reset_state[10:12] = nut_pos
        self.env.env.reset_to({"states": reset_state})

        nut_pos = self.env.env.env.sim.data.body_xpos[
            self.env.env.env.obj_body_id["SquareNut"]
        ]
        peg_pos1 = np.array(
            self.env.env.env.sim.data.body_xpos[self.env.env.env.peg1_body_id]
        )
        peg_pos2 = np.array(
            self.env.env.env.sim.data.body_xpos[self.env.env.env.peg2_body_id]
        )

        if np.linalg.norm(nut_pos - peg_pos1) < np.linalg.norm(nut_pos - peg_pos2):
            self.target_peg_id = 1
        else:
            self.target_peg_id = 0

        # return obs
        obs = self.get_observation()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # 4-stage long-horizon reward (peg_id = 0 throughout, as in LDP).
        peg_id = 0
        peg_pos = self.env.env.env.sim.data.body_xpos[
            self.env.env.env.obj_body_id["SquareNut"]
        ]
        on_peg = self.env.env.env.on_peg(peg_pos, peg_id)

        on_saddle_drop = (
            np.linalg.norm(obs["robot0_eef_pos"] - np.array([0, 0, 0.89])) < 0.03
        )
        on_saddle = (
            np.linalg.norm(obs["robot0_eef_pos"] - np.array([0, 0, 0.89 + 0.2]))
            < 0.09
        )

        if self.rew == 0 and on_peg:
            self.rew += 1
        elif self.rew == 1 and on_saddle:
            self.rew += 1
        elif self.rew == 2 and on_peg:
            self.rew += 1
        elif self.rew == 3 and on_saddle_drop:
            self.rew += 1

        reward = 1 if self.rew == 4 else 0

        return obs, reward, done, info

    def render(self, mode="rgb_array"):
        return self.env.render(mode=mode)
