from typing import Union
import torch
import numpy as np
import functools


# ── Pure-PyTorch rotation conversions (replaces pytorch3d.transforms) ────

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rotation vector) to 3×3 rotation matrix (Rodrigues)."""
    angle = axis_angle.norm(dim=-1, keepdim=True).unsqueeze(-1)  # (..., 1, 1)
    axis = axis_angle / (axis_angle.norm(dim=-1, keepdim=True) + 1e-8)  # (..., 3)

    K = torch.zeros(axis.shape[:-1] + (3, 3), device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]

    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand_as(K)
    R = eye + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    return R


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3×3 rotation matrix to axis-angle via quaternion intermediate."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3×3 rotation matrix to quaternion (w, x, y, z)."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {matrix.shape}")

    batch_shape = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)
    B = m.shape[0]

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(B, 4, device=matrix.device, dtype=matrix.dtype)

    s = torch.sqrt(torch.clamp(trace + 1, min=1e-10)) * 2  # s = 4*w
    quat[:, 0] = 0.25 * s
    quat[:, 1] = (m[:, 2, 1] - m[:, 1, 2]) / s
    quat[:, 2] = (m[:, 0, 2] - m[:, 2, 0]) / s
    quat[:, 3] = (m[:, 1, 0] - m[:, 0, 1]) / s

    # Handle cases where trace is not the largest diagonal
    cond1 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]) & (trace <= 0)
    s1 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-10)) * 2
    quat1 = torch.stack([
        (m[:, 2, 1] - m[:, 1, 2]) / s1,
        0.25 * s1,
        (m[:, 0, 1] + m[:, 1, 0]) / s1,
        (m[:, 0, 2] + m[:, 2, 0]) / s1,
    ], dim=-1)

    cond2 = (m[:, 1, 1] > m[:, 2, 2]) & ~cond1 & (trace <= 0)
    s2 = torch.sqrt(torch.clamp(1.0 + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], min=1e-10)) * 2
    quat2 = torch.stack([
        (m[:, 0, 2] - m[:, 2, 0]) / s2,
        (m[:, 0, 1] + m[:, 1, 0]) / s2,
        0.25 * s2,
        (m[:, 1, 2] + m[:, 2, 1]) / s2,
    ], dim=-1)

    cond3 = ~cond1 & ~cond2 & (trace <= 0)
    s3 = torch.sqrt(torch.clamp(1.0 + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], min=1e-10)) * 2
    quat3 = torch.stack([
        (m[:, 1, 0] - m[:, 0, 1]) / s3,
        (m[:, 0, 2] + m[:, 2, 0]) / s3,
        (m[:, 1, 2] + m[:, 2, 1]) / s3,
        0.25 * s3,
    ], dim=-1)

    quat = torch.where(cond1.unsqueeze(-1), quat1, quat)
    quat = torch.where(cond2.unsqueeze(-1), quat2, quat)
    quat = torch.where(cond3.unsqueeze(-1), quat3, quat)

    return quat.reshape(batch_shape + (4,))


def quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 3×3 rotation matrix."""
    q = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))

    return R


def quaternion_to_axis_angle(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to axis-angle."""
    q = quaternion / (quaternion.norm(dim=-1, keepdim=True) + 1e-8)
    # Ensure w > 0 for consistent representation
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., 0:1]
    xyz = q[..., 1:4]
    sin_half = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / (sin_half + 1e-8)
    return axis * angle


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Extract first two columns of rotation matrix as 6D representation."""
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3×3 matrix via Gram-Schmidt."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]

    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-2)


# Bundle into a namespace that mirrors pytorch3d.transforms
class _RotationFunctions:
    axis_angle_to_matrix = staticmethod(axis_angle_to_matrix)
    matrix_to_axis_angle = staticmethod(matrix_to_axis_angle)
    matrix_to_quaternion = staticmethod(matrix_to_quaternion)
    quaternion_to_matrix = staticmethod(quaternion_to_matrix)
    matrix_to_rotation_6d = staticmethod(matrix_to_rotation_6d)
    rotation_6d_to_matrix = staticmethod(rotation_6d_to_matrix)

pt = _RotationFunctions()


# ── RotationTransformer (unchanged interface) ────────────────────────────

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
            from_rep='axis_angle',
            to_rep='rotation_6d',
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix
