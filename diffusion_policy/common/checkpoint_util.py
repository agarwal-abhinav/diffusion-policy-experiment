from typing import Optional, Dict, List
import os


class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt',
            min_training_steps: int = 0,
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.min_training_steps = min_training_steps
        self.path_value_map = dict()

    def get_ckpt_path(
            self,
            data: Dict[str, float],
            protected_ckpts: set=set()
        ) -> Optional[str]:
        """
        Args:
            data: a dictionary containing the monitoring values.
                Must include the monitor_key. If min_training_steps > 0,
                must also include 'global_step'.
            protected_ckpts: a list of paths that should not be deleted
                These paths are usually ckpts being tracked by other TopKCheckpointManagers
        Returns:
            ckpt_path: the path to the checkpoint to save, or None if no checkpoint should be saved
        """
        if self.k == 0:
            return None

        # Skip if below minimum training steps
        if self.min_training_steps > 0:
            current_step = data.get('global_step', 0)
            if current_step < self.min_training_steps:
                return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))

        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path

        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path) and delete_path not in protected_ckpts:
                os.remove(delete_path)
            return ckpt_path

    def get_path_value_map(self):
        return self.path_value_map

    def state_dict(self):
        return {
            'path_value_map': dict(self.path_value_map),
        }

    def load_state_dict(self, state_dict):
        self.path_value_map = dict(state_dict.get('path_value_map', {}))


class IntervalCheckpointManager:
    """
    Saves evenly-spaced checkpoints over the final portion of training.

    For example, with total_training_steps=200000, last_n_steps=30000,
    num_checkpoints=3: saves at steps 180000, 190000, 200000.
    """

    def __init__(self,
            save_dir,
            total_training_steps: int,
            last_n_steps: int = 30000,
            num_checkpoints: int = 3,
            format_str='step={global_step:06d}-interval.ckpt',
        ):
        self.save_dir = save_dir
        self.total_training_steps = total_training_steps
        self.last_n_steps = last_n_steps
        self.num_checkpoints = num_checkpoints
        self.format_str = format_str
        self.saved_paths = set()

        # Pre-compute the steps at which to save
        start_step = max(0, total_training_steps - last_n_steps)
        if num_checkpoints <= 1:
            self.save_steps = {total_training_steps}
        else:
            interval = last_n_steps / (num_checkpoints - 1)
            self.save_steps = set()
            for i in range(num_checkpoints):
                step = int(start_step + i * interval)
                self.save_steps.add(step)

    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        """
        Args:
            data: must include 'global_step'
        Returns:
            ckpt_path if this step should be saved, else None
        """
        current_step = data.get('global_step', 0)

        # Find the closest save_step that hasn't been saved yet
        # and is <= current_step
        best_step = None
        for step in self.save_steps:
            if step <= current_step and step not in self.saved_paths:
                if best_step is None or step > best_step:
                    best_step = step

        if best_step is None:
            return None

        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        self.saved_paths.add(best_step)
        return ckpt_path

    def get_all_paths(self):
        """Return paths tracked by this manager (for protected_ckpts)."""
        return self.saved_paths

    def state_dict(self):
        return {
            'saved_paths': set(self.saved_paths),
        }

    def load_state_dict(self, state_dict):
        self.saved_paths = set(state_dict.get('saved_paths', set()))


class CheckpointManagers:
    """
    Wrapper holding all checkpoint managers with state persistence.
    Stored on the workspace as self.checkpoint_managers so that
    BaseWorkspace.save_checkpoint auto-saves/loads it.
    """

    def __init__(self,
            topk_managers: List[TopKCheckpointManager],
            interval_manager: Optional[IntervalCheckpointManager] = None,
        ):
        self.topk_managers = topk_managers
        self.interval_manager = interval_manager

    def state_dict(self):
        state = {
            'topk': [m.state_dict() for m in self.topk_managers],
        }
        if self.interval_manager is not None:
            state['interval'] = self.interval_manager.state_dict()
        return state

    def load_state_dict(self, state_dict):
        topk_states = state_dict.get('topk', [])
        for manager, mstate in zip(self.topk_managers, topk_states):
            manager.load_state_dict(mstate)
        if self.interval_manager is not None and 'interval' in state_dict:
            self.interval_manager.load_state_dict(state_dict['interval'])
