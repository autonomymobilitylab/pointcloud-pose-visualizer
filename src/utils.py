import numpy as np
from scipy.spatial.transform import Rotation


def rot2hom(rot: np.ndarray | Rotation) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix() if isinstance(rot, Rotation) else rot
    return T


def rot_transl2hom(rot: np.ndarray | Rotation, transl: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix and a 3D translation vector to a 4x4 homogeneous transformation matrix.
    """
    T = rot2hom(rot)
    T[:3, 3] = transl
    return T


def hom2rot(T: np.ndarray) -> np.ndarray:
    """
    Extract the 3x3 rotation matrix from a 4x4 homogeneous transformation matrix.
    """
    return T[:3, :3]


def hom2transl(T: np.ndarray) -> np.ndarray:
    """
    Extract the 3D translation vector from a 4x4 homogeneous transformation matrix.
    """
    return T[:3, 3]
