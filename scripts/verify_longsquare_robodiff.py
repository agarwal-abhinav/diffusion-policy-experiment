"""
Step-0 verification for the long-square port.

Run this in the `robodiff` env. It checks:
  1. NutAssemblySquare can be instantiated from a robomimic env_meta in robodiff.
  2. The mujoco state vector matches what the MultimodalSquareWrapper hardcodes
     (45 elements), so that `reset_state[10:12] = nut_pos` writes to the right
     indices.
  3. The robosuite NutAssembly env exposes `peg1_body_id`, `peg2_body_id`,
     `obj_body_id['SquareNut']`, and `on_peg(...)`, all of which the wrapper
     accesses directly.
  4. The wrapper itself (imported from the cloned ldp repo) can be wrapped
     around the robomimic env, can `reset()`, and can take a few `step(action)`
     calls without raising.

If any check fails, the script prints which one and bails. If all pass, the
long-square port should be safe to do against `robodiff`'s robosuite==1.2.0
without needing to install cheng-chi's robosuite fork.
"""
import sys
import os
import json
import numpy as np
import h5py


SQUARE_HDF5 = "/home/abhinav/RLG_new/diffusion-policy-experiment/data/robomimic/square/ph/image_abs.hdf5"
LDP_REPO = "/home/abhinav/external/ldp"
# We need ldp's MultimodalSquareWrapper, which lives inside the longhist runner.
# To import it without dragging in their HSIC dependencies, we add the ldp
# repo to sys.path and import directly.
sys.path.insert(0, LDP_REPO)


def step1_env_instantiates():
    """Can we create the env from the square hdf5?"""
    import robomimic.utils.file_utils as FU
    import robomimic.utils.env_utils as EU
    import robomimic.utils.obs_utils as OU

    env_meta = FU.get_env_metadata_from_dataset(SQUARE_HDF5)
    env_meta["env_kwargs"]["use_object_obs"] = False
    # match the wrapper's abs_action setup
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

    OU.initialize_obs_modality_mapping_from_dict({
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    })
    env = EU.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return env


def step2_state_shape(env):
    """Does the mujoco state vector match the hardcoded 45-element reset_state?"""
    state = env.env.sim.get_state().flatten()
    print(f"  mujoco state.flatten().shape = {state.shape}")
    if state.shape[0] != 45:
        return False, f"state shape {state.shape[0]} != 45 (wrapper's hardcoded length)"
    return True, "state shape matches wrapper's reset_state (45)"


def step3_required_attrs(env):
    """All attrs the wrapper accesses on env.env.env."""
    rs = env.env  # this is the robosuite NutAssembly env
    checks = {}
    for name in ["peg1_body_id", "peg2_body_id"]:
        checks[name] = hasattr(rs, name)
    checks["obj_body_id"] = hasattr(rs, "obj_body_id")
    checks["obj_body_id_has_SquareNut"] = (
        checks["obj_body_id"] and "SquareNut" in rs.obj_body_id
    )
    checks["on_peg"] = callable(getattr(rs, "on_peg", None))
    for k, v in checks.items():
        print(f"  {k}: {v}")
    ok = all(checks.values())
    return ok, ("all required attrs present" if ok else f"missing: "
                + ", ".join(k for k, v in checks.items() if not v))


def step4_wrapper_reset_step():
    """Try the MultimodalSquareWrapper end-to-end on a couple of steps."""
    import robomimic.utils.file_utils as FU
    import robomimic.utils.env_utils as EU
    import robomimic.utils.obs_utils as OU
    from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper

    # The wrapper class is defined inline inside the longhist runner.
    # We can't `from ... import MultimodalSquareWrapper` directly because
    # that module imports the hsic/mlp_correlation files which require extra
    # deps. Import the class by `runpy`-ing the source up to the class def.
    from importlib.util import spec_from_file_location, module_from_spec
    wrapper_src = os.path.join(LDP_REPO, "diffusion_policy/env_runner/robomimic_longhist_image_runner.py")
    # Read source, isolate just the imports + MultimodalSquareWrapper class.
    with open(wrapper_src) as f:
        src = f.read()
    head = src.split("class RobomimicImageRunner")[0]
    # Strip imports that we don't need (and that would fail without extra deps).
    safe_lines = []
    for line in head.splitlines():
        if any(bad in line for bad in [
            "from hsic", "from mlp_correlation", "import wandb",
        ]):
            continue
        safe_lines.append(line)
    safe_head = "\n".join(safe_lines)
    ns = {"__name__": "_lsq_wrapper_iso", "__file__": wrapper_src}
    exec(compile(safe_head, wrapper_src, "exec"), ns)
    MultimodalSquareWrapper = ns["MultimodalSquareWrapper"]

    shape_meta = {
        "action": {"shape": [10]},
        "obs": {
            "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
        },
    }

    env_meta = FU.get_env_metadata_from_dataset(SQUARE_HDF5)
    env_meta["env_kwargs"]["use_object_obs"] = False
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
    OU.initialize_obs_modality_mapping_from_dict({
        "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
    })
    base_env = EU.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=True, use_image_obs=True,
    )
    base_env.env.hard_reset = False

    wrapped = MultimodalSquareWrapper(RobomimicImageWrapper(
        env=base_env, shape_meta=shape_meta, init_state=None,
        render_obs_key="agentview_image",
    ))
    wrapped.seed(0)
    obs = wrapped.reset()
    print(f"  reset OK. obs keys = {sorted(obs.keys())}")

    # take 5 random actions. The wrapper passes action straight through to
    # the underlying robosuite env, which has action_dim=7 (3 pos + 3 axis-angle
    # rot + 1 gripper). During real eval the runner calls undo_transform_action
    # to convert rotation_6d (dim 10) -> axis_angle (dim 7) before stepping;
    # here we just pass raw 7-dim actions since we are not running a policy.
    rng = np.random.default_rng(0)
    # Step through the wrapper chain with raw env-native action dim
    # (3 pos + 3 axis-angle rot + 1 gripper = 7). In the real env_runner this
    # is produced by `undo_transform_action`; here we just generate it directly.
    # robosuite expects action_dim from its base env (env.env.env.action_dim).
    raw_action_dim = wrapped.env.env.env.action_dim
    print(f"  wrapped.action_space.shape = {wrapped.action_space.shape}  "
          f"(transformed; runner calls undo_transform_action to {raw_action_dim}-dim)")
    print(f"  underlying robosuite action_dim = {raw_action_dim}")
    for t in range(5):
        action = rng.uniform(-0.1, 0.1, size=(raw_action_dim,)).astype(np.float32)
        obs, reward, done, info = wrapped.step(action)
        print(f"  step {t}: reward={reward}, done={done}, rew_stage={wrapped.rew}")
    return True, "wrapper reset + 5 steps OK"


def main():
    print("=" * 70)
    print("Step 0 verification for the long-square port")
    print("Running with python:", sys.executable)
    import robomimic, robosuite
    print(f"  robomimic {robomimic.__version__}, robosuite {robosuite.__version__}")
    print("=" * 70)

    try:
        env = step1_env_instantiates()
        print("[1] env_instantiates: OK")
    except Exception as e:
        print(f"[1] env_instantiates: FAILED — {type(e).__name__}: {e}")
        sys.exit(1)

    ok, msg = step2_state_shape(env)
    print(f"[2] state_shape: {'OK' if ok else 'FAILED'} — {msg}")
    state_ok = ok

    ok, msg = step3_required_attrs(env)
    print(f"[3] required_attrs: {'OK' if ok else 'FAILED'} — {msg}")
    attrs_ok = ok

    try:
        ok, msg = step4_wrapper_reset_step()
        print(f"[4] wrapper_reset_step: {'OK' if ok else 'FAILED'} — {msg}")
        wrapper_ok = ok
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[4] wrapper_reset_step: FAILED — {type(e).__name__}: {e}")
        wrapper_ok = False

    print("=" * 70)
    overall = state_ok and attrs_ok and wrapper_ok
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    if not overall:
        sys.exit(2)


if __name__ == "__main__":
    main()
