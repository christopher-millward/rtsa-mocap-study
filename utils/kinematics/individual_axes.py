
"""Functions for calculating the amount of rotation about each axis.

Author: Christopher Millward
"""

import numpy as np
import numpy.typing as npt
from typing import Tuple
from scipy.spatial.transform import Rotation as R
from utils.kinematics.cumulative import create_rotation_matrices


def _validate_orthonorm_and_det(matrices: npt.NDArray[np.float64]) -> None:
    """Validate that a batch of 3x3 matrices are proper rotation matrices.

    The check is fully vectorized over the batch dimension. It verifies that
    each matrix is orthonormal by confirming ``R.T @ R == I`` for every matrix
    in the batch, and it rejects improper rotations by requiring each
    determinant to be approximately ``+1``.

    Args:
        matrices (npt.NDArray[np.float64]): Array of candidate rotation
            matrices with shape ``(n_steps, 3, 3)``.

    Raises:
        ValueError: If the input is not a batch of 3x3 matrices, if any matrix
            is not orthonormal, or if any determinant differs from ``+1``.
    """
    gram = np.matmul(np.transpose(matrices, (0, 2, 1)), matrices)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)
    if not np.allclose(gram, identity, atol=1e-8):
        raise ValueError("relative_rotations must be orthonormal rotation matrices")

    dets = np.linalg.det(matrices)
    if not np.allclose(dets, 1.0, atol=1e-6):
        raise ValueError("relative_rotations must have determinant 1")


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
        ValueError: If input does not have shape (n_frames, 3, 3), has fewer
            than two frames, is not orthonormal, or has determinants that differ from 1.
    """
    
    # Coerce to ndarray with correct dtype
    matrices = np.asarray(rotation_matrices, dtype=np.float64)

    # Validate shape is (n_frames, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            'rotation_matrices must have shape (n_frames, 3, 3)'
        )

    # Validate shape has at least 2 frames
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


    # Validate shape is (n_frames, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("relative_rotations must have shape (n_steps, 3, 3)")
    
    # Reject empty batch explicitly
    if matrices.shape[0] == 0:
        raise ValueError("relative_rotations must contain at least one matrix")

    #Validate orthonormality and determinant of each matrix
    _validate_orthonorm_and_det(matrices)

    euler_angles = R.from_matrix(matrices).as_euler(
        seq="YXY",  # Must be uppercase for intrinsic rotations in scipy
        degrees=False,
    )

    return np.asarray(euler_angles, dtype=np.float64)


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
        ValueError: If input contains negative values.
        ValueError: If input is empty.
    """
    
    # Ensure correct dtype
    all_components = np.asarray(euler_angles, dtype=np.float64)

    # Validate shape is 2D with exactly 3 columns
    if all_components.ndim != 2 or all_components.shape[1] != 3:
        raise ValueError("euler_angles must have shape (n_steps, 3)")

    # Reject empty batch
    if all_components.shape[0] == 0:
        raise ValueError("euler_angles must contain at least one row")

    # Reject negative values (tests expect rejection)
    if np.any(all_components < 0.0):
        raise ValueError("euler_angles must not contain negative values")

    # Sum each column (component) across rows and convert to native floats
    sums = np.sum(all_components, axis=0)
    return (float(sums[0]), float(sums[1]), float(sums[2]))


def calculate_cumulative_axis_motion(
    data: npt.NDArray[np.float64],
    arm: str,
) -> Tuple[float, float, float]:
    """Calculate cumulative Y-X-Y component motion for one arm.

    This is the high-level orchestration function for one arm. It creates relative 
    rotation matrices from the absolute orientations, decomposes them into Euler
    components, and then sums each component independently to get cumulative motion
    values.

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

    # Validate arm
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    # Validate data shape
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError(
            'Data must be a 2D array with exactly 18 columns.'
        )

    # Validate data is not empty
    if data_array.shape[0] == 0:
        raise ValueError("Input data cannot be empty")

    # Create absolute rotation matrices for the specified arm
    matrices = create_rotation_matrices(data_array, arm)

    # Compute relative rotation matrices
    relative_rotations = compute_incremental_rotation_matrices(matrices)

    # Decompose relative rotations into Euler angles
    euler_angles = decompose_rotation_matrices_yxy(relative_rotations)

    # Accumulate Euler components independently
    cumulative_components = accumulate_euler_components(euler_angles)

    return cumulative_components


# def calculate_cumulative_axis_motion_both_arms(
#     data: npt.NDArray[np.float64],
# ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
#     """Calculate cumulative Y-X-Y component motion for left and right arms.

#     This convenience wrapper calls the single-arm cumulative function for both
#     arms and returns both results in one structure.

#     Args:
#         data (npt.NDArray[np.float64]): Motion capture table loaded via
#             ``np.loadtxt`` with at least 18 columns.

#     Returns:
#         Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
#             ``(left_components, right_components)``, where each inner tuple is
#             the cumulative Y-X-Y component sums in radians.

#     Raises:
#         ValueError: If input validation fails for either arm.
#     """
#     # Call calculate_cumulative_axis_motion for arm='L'.
#     # Call calculate_cumulative_axis_motion for arm='R'.
#     # Return both tuples as (left_components, right_components).
#     raise NotImplementedError

