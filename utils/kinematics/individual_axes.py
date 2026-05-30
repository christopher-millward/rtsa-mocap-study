
"""Functions for calculating the amount of rotation about each axis.

Author: Christopher Millward
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def compute_incremental_rotation_matrices(
    rotation_matrices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build per-timestep relative rotation matrices from absolute orientations.

    For each timestep t > 0, this function will compute the relative rotation
    that maps the previous frame orientation to the current frame orientation.
    The intended vectorized operation is:
        R_delta[t] = R_current[t] @ R_previous[t].T

    Args:
        rotation_matrices (npt.NDArray[np.float64]): Absolute rotation matrices
            for one arm with shape (n_frames, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Relative rotation matrices for each transition
            with shape (n_frames - 1, 3, 3).

    Raises:
        ValueError: If input does not have shape (n_frames, 3, 3) or has fewer
            than two frames.
    """
    
    # Coerce to ndarray with correct dtype
    matrices = np.asarray(rotation_matrices, dtype=np.float64)

    # Validate shape: must be (n_frames, 3, 3) and have at least two frames
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            'rotation_matrices must have shape (n_frames, 3, 3)'
        )

    n_frames = matrices.shape[0]
    if n_frames < 2:
        raise ValueError(
            'rotation_matrices must contain at least two frames to compute increments'
        )

    # Previous and current frames (vectorized stacks)
    prev = matrices[:-1]
    curr = matrices[1:]

    # Compute relative rotations: R_delta[t] = R_current[t] @ R_previous[t].T
    deltas = np.matmul(curr, np.transpose(prev, (0, 2, 1))) # keep batch axis fixed, swap matrix axes 

    return deltas


def decompose_rotation_matrices_yxy(
    relative_rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Decompose relative rotations into Euler components using Y-X-Y sequence.

    This function will perform a batch decomposition of each 3x3 relative
    rotation matrix into three Euler angles following the Y-X-Y convention,
    returning one angle triplet per timestep transition.

    Args:
        relative_rotations (npt.NDArray[np.float64]): Relative rotation matrices
            with shape (n_steps, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Euler angles in radians with shape
            (n_steps, 3), ordered as (first_Y, X, second_Y).

    Raises:
        ValueError: If the input shape is invalid.
    """
    # Coerce to ndarray with correct dtype
    matrices = np.asarray(relative_rotations, dtype=np.float64)

    # Validate shape: must be (n_steps, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            "relative_rotations must have shape (n_steps, 3, 3)"
        )

    n_steps = matrices.shape[0]
    # Empty batch -> empty result
    if n_steps == 0:
        raise ValueError(
            "relative_rotations must contain at least one step for decomposition"
        )

    # Perform vectorized Euler decomposition using intrinsic Y-X-Y sequence
    euler_angles = R.from_matrix(matrices).as_euler(
        seq="yxy",
        degrees=False,
    )

    return np.asarray(euler_angles, dtype=np.float64)

# def decompose_rotation_matrices_yxy(
#     relative_rotations: npt.NDArray[np.float64],
# ) -> npt.NDArray[np.float64]:
#     """Decompose relative rotations into Euler components using Y-X-Y sequence.

#     This function will perform a batch decomposition of each 3x3 relative
#     rotation matrix into three Euler angles following the Y-X-Y convention,
#     returning one angle triplet per timestep transition.

#     Args:
#         relative_rotations (npt.NDArray[np.float64]): Relative rotation matrices
#             with shape (n_steps, 3, 3).

#     Returns:
#         npt.NDArray[np.float64]: Euler angles in radians with shape
#             (n_steps, 3), ordered as (first_Y, X, second_Y).

#     Raises:
#         ValueError: If the input shape is invalid.
#     """
#     # Coerce to ndarray with correct dtype
#     matrices = np.asarray(relative_rotations, dtype=np.float64)

#     # Validate shape: must be (n_steps, 3, 3)
#     if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
#         raise ValueError("relative_rotations must have shape (n_steps, 3, 3)")

#     n_steps = matrices.shape[0]
#     # Empty batch -> empty result
#     if n_steps == 0:
#         raise ValueError("relative_rotations must contain at least one step for decomposition")

#     # Sanity-check: each matrix should be a valid rotation (orthonormal, det ~ 1)
#     # Use a reasonably strict tolerance for orthonormality and determinant.
#     rt_r = np.einsum("nij,njk->nik", np.transpose(matrices, (0, 2, 1)), matrices)
#     identity = np.broadcast_to(np.eye(3, dtype=np.float64), rt_r.shape)
#     ortho_ok = np.allclose(rt_r, identity, atol=1e-8)
#     dets = np.linalg.det(matrices)
#     det_ok = np.allclose(dets, 1.0, atol=1e-6)
#     if not (ortho_ok and det_ok):
#         raise ValueError("relative_rotations must be valid rotation matrices")

#     # Extract elements used in the Y-X-Y extraction formulas
#     r = matrices
#     # middle angle b: cos(b) = R[1,1]
#     cb = np.clip(r[:, 1, 1], -1.0, 1.0)
#     b = np.arccos(cb)

#     # sin(b) magnitude from R[1,0] and R[1,2]
#     sb = np.hypot(r[:, 1, 0], r[:, 1, 2])

#     # Prepare arrays for a (first Y) and c (second Y)
#     a = np.empty(n_steps, dtype=np.float64)
#     c = np.empty(n_steps, dtype=np.float64)

#     # Non-singular case: sb > eps
#     eps = 1e-12
#     nonsing = sb > eps
#     if np.any(nonsing):
#         # a = atan2(R[0,1], R[2,1]) ; c = atan2(R[1,0], -R[1,2])
#         a[nonsing] = np.arctan2(r[nonsing, 0, 1], r[nonsing, 2, 1])
#         c[nonsing] = np.arctan2(r[nonsing, 1, 0], -r[nonsing, 1, 2])

#     # Singular case: sb ~= 0 -> b is approximately 0 or pi
#     if np.any(~nonsing):
#         idx = np.where(~nonsing)[0]
#         for i in idx:
#             # If cb > 0 -> b ~ 0: Rx(0) -> combined rotation about Y by (a + c)
#             if cb[i] > 0:
#                 summed = np.arctan2(r[i, 0, 2], r[i, 0, 0])
#                 a[i] = summed
#                 b[i] = 0.0
#                 c[i] = 0.0
#             else:
#                 # b ~ pi: choose a representation that keeps the result deterministic.
#                 # Use the relation for (a - c) from the matrix elements and set c=0.
#                 summed = np.arctan2(-r[i, 0, 2], r[i, 0, 0])
#                 a[i] = summed
#                 b[i] = np.pi
#                 c[i] = 0.0

#     angles = np.stack((a, b, c), axis=1)
#     return angles


def accumulate_euler_components(
    euler_angles: npt.NDArray[np.float64],
) -> Tuple[float, float, float]:
    """Sum Euler components independently across all timestep transitions.

    Given a matrix of decomposed Euler angles, this function will sum each
    component independently to produce cumulative motion values for the three
    sequence positions.

    Args:
        euler_angles (npt.NDArray[np.float64]): Array of Euler angles with shape
            (n_steps, 3).

    Returns:
        Tuple[float, float, float]: Cumulative sums of the first, second, and
            third Euler components, respectively.

    Raises:
        ValueError: If input does not have shape (n_steps, 3).
    """
    # Validate input as a 2D array with exactly three columns.
    # Sum each Euler component independently across rows.
    # Convert summed values to Python floats and return as a 3-tuple.
    raise NotImplementedError


def calculate_cumulative_axis_motion(
    data: npt.NDArray[np.float64],
    arm: str,
) -> Tuple[float, float, float]:
    """Calculate cumulative Y-X-Y component motion for one arm.

    This is the high-level orchestration function for one arm. It is designed
    to reuse existing matrix extraction functionality and then apply the
    timestep-relative/decomposition/summation pipeline.

    Args:
        data (npt.NDArray[np.float64]): Motion capture table loaded via
            ``np.loadtxt`` with at least 18 columns.
        arm (str): Arm identifier, either 'L' or 'R'.

    Returns:
        Tuple[float, float, float]: Cumulative sums for the three Y-X-Y Euler
            components in radians.

    Raises:
        ValueError: If input data is malformed or arm is invalid.

    Reuse:
        - Reuse ``create_rotation_matrices`` to extract 3x3 matrices per frame.
    """
    # Reuse create_rotation_matrices(data, arm) to get absolute orientations.
    # Compute relative rotations between consecutive frames.
    # Decompose relative rotations with Y-X-Y convention.
    # Sum each Euler component independently.
    # Return cumulative component tuple.
    raise NotImplementedError


def calculate_cumulative_axis_motion_both_arms(
    data: npt.NDArray[np.float64],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Calculate cumulative Y-X-Y component motion for left and right arms.

    This convenience wrapper calls the single-arm cumulative function for both
    arms and returns both results in one structure.

    Args:
        data (npt.NDArray[np.float64]): Motion capture table loaded via
            ``np.loadtxt`` with at least 18 columns.

    Returns:
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
            ``(left_components, right_components)``, where each inner tuple is
            the cumulative Y-X-Y component sums in radians.

    Raises:
        ValueError: If input validation fails for either arm.
    """
    # Call calculate_cumulative_axis_motion for arm='L'.
    # Call calculate_cumulative_axis_motion for arm='R'.
    # Return both tuples as (left_components, right_components).
    raise NotImplementedError

