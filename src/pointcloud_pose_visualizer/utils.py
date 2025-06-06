import math
from typing import Literal

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


def roteuler2rotmat(
    rot_x: float, rot_y: float, rot_z: float, units: Literal["deg", "rad"] = "rad"
) -> np.ndarray[float]:
    """
    Convert Euler angles into a 3x3 rotation matrix.
    The rotation is applied in the order of X, Y, Z.

    Args:
        rot_x: Rotation around the x-axis in radians.
        rot_y: Rotation around the y-axis in radians.
        rot_z: Rotation around the z-axis in radians.
    Returns:
        A 3x3 rotation matrix.
    """

    if units == "deg":
        rot_x = math.radians(rot_x)
        rot_y = math.radians(rot_y)
        rot_z = math.radians(rot_z)
    elif units != "rad":
        raise ValueError("Units must be 'rad' or 'deg'.")

    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(rot_x), -math.sin(rot_x)],
            [0, math.sin(rot_x), math.cos(rot_x)],
        ]
    )
    Ry = np.array(
        [
            [math.cos(rot_y), 0, math.sin(rot_y)],
            [0, 1, 0],
            [-math.sin(rot_y), 0, math.cos(rot_y)],
        ]
    )
    Rz = np.array(
        [
            [math.cos(rot_z), -math.sin(rot_z), 0],
            [math.sin(rot_z), math.cos(rot_z), 0],
            [0, 0, 1],
        ]
    )
    # Combine rotations: order X, Y, Z.
    return Rz @ Ry @ Rx
