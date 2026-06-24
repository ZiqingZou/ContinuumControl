import numpy as np
from scipy.spatial.transform import Rotation as R


def rotmat_to_angle_axes(rotm):
    """
    Convert rotation matrix or batch of rotation matrices to (angle, axis).

    Args:
        rotm (np.ndarray): Rotation matrix with shape (3,3) or batch shape (N,3,3).

    Returns:
        angle_axes (np.ndarray): Each row contains [angle (rad), axis_x, axis_y, axis_z]. 
                                 For near-zero rotations (angle <= eps) the axis is [0,0,0].
    """

    eps = 1e-8

    rotm = np.asarray(rotm)
    if rotm.ndim == 2:
        rotm = rotm[None, ...]                # make batch (1,3,3)

    # build Rotation and get rotation vectors (axis * angle)
    Robj = R.from_matrix(rotm)                # accepts (N,3,3)
    rotvec = Robj.as_rotvec()                 # shape (N,3)

    angles = np.linalg.norm(rotvec, axis=1)   # angle = ||rotvec||
    axes = np.zeros_like(rotvec)
    mask = angles > eps
    axes[mask] = rotvec[mask] / angles[mask][:, None]
    angle_axes = np.concatenate([angles[:, None], axes], axis=1)  # shape (N,4)
    return angle_axes


def rotvec_to_angle_axes(rotvec):
    """
    Convert rotation vector or batch of rotation vectors to (angle, axis).

    Args:
        rotvec (np.ndarray): Rotation vector with shape (3,) or batch shape (N,3).

    Returns:
        angle_axes (np.ndarray): Each row contains [angle (rad), axis_x, axis_y, axis_z]. 
                                 For near-zero rotations (angle <= eps) the axis is [0,0,0].
    """

    eps = 1e-8

    rotvec = np.asarray(rotvec)
    if rotvec.ndim == 1:
        rotvec = rotvec[None, ...]                # make batch (1,3)

    angles = np.linalg.norm(rotvec, axis=1)   # angle = ||rotvec||
    axes = np.zeros_like(rotvec)
    mask = angles > eps
    axes[mask] = rotvec[mask] / angles[mask][:, None]
    angle_axes = np.concatenate([angles[:, None], axes], axis=1)  # shape (N,4)
    return angle_axes
