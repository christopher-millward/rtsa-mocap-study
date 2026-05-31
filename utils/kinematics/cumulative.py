
"""Functions for calculating the cumulative rotation across all axes .

Author: Christopher Millward
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt


def create_rotation_matrices(data: npt.NDArray[np.float64], arm: str) -> npt.NDArray[np.float64]:
    """Extract a batch of 3x3 rotation matrices for a specified arm.

    The function operates on a 2D motion-capture array and slices the nine
    rotation values for the requested arm from every frame. Those nine values
    are reshaped into a 3x3 matrix for each frame, producing a vectorized
    stack of rotation matrices.

    Args:
        data (npt.NDArray[np.float64]): A 2D array with at least 18 columns,
            where columns 0-8 contain left arm rotation data and columns
            9-17 contain right arm rotation data.
        arm (str): Arm identifier, either 'L' (left) or 'R' (right).

    Returns:
        npt.NDArray[np.float64]: An array of 3x3 rotation matrices with shape
            (n_frames, 3, 3).

    Raises:
        ValueError: If arm is not 'L' or 'R'.
        ValueError: If the row does not contain enough values.
    """
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError(
            'Data must be a 2D array with exactly 18 columns.'
        )

    start_index = 0 if arm == 'L' else 9
    return data_array[:, start_index:start_index + 9].reshape(-1, 3, 3)


def calculate_rotation_angles(rotation_matrices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate rotation angles in radians for a batch of rotation matrices.

    Computes the angle of rotation for each 3x3 rotation matrix using the
    trace formula: θ = arccos((trace(R) - 1) / 2). The implementation is
    vectorized across the full batch of matrices.

    Args:
        rotation_matrices (npt.NDArray[np.float64]): An array of rotation
            matrices with shape (n_frames, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Rotation angles in radians for each frame,
            with shape (n_frames,).

    Notes:
        - The result is always positive and in the range [0, π].
        - Due to numerical precision, trace values outside [-1, 3] are
          clamped to a valid range before computing arccos.

    Example:
        >>> R = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        >>> angles = calculate_rotation_angles(R)
        >>> np.isclose(angles[0], 0)
        True
    """

    traces = np.trace(rotation_matrices, axis1=1, axis2=2)
    cos_angles = np.clip((traces - 1) / 2, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return angles


def calculate_total_rotation(data: npt.NDArray[np.float64], arm: str) -> float:
    """Calculate total rotation magnitude for an arm across all frames.

    Extracts a batch of rotation matrices using `create_rotation_matrices`,
    computes a vector of rotation angles with `calculate_rotation_angles`,
    and sums the result to get total accumulated rotation.

    Args:
        data (npt.NDArray[np.float64]): A 2D array with 18 numeric columns per row,
            containing motion capture rotation data for both arms.
        arm (str): Arm identifier, either 'L' (left) or 'R' (right).

    Returns:
        float: Total accumulated rotation in radians.

    Raises:
        ValueError: If arm is not 'L' or 'R'.
        ValueError: If the array does not have the expected shape.
    """
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError(
            'Data must be a 2D array with exactly 18 columns.'
        )

    matrices = create_rotation_matrices(data_array, arm)
    angles = calculate_rotation_angles(matrices)
    
    return float(angles.sum())


def calculate_arm_rotations(data: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """Calculate total rotation for both arms from a motion capture array.

    Calls `calculate_total_rotation` for each arm, which internally uses
    `create_rotation_matrices` and `calculate_rotation_angles` to compute
    the rotation angles for each frame in a vectorized batch.

    Args:
        data (npt.NDArray[np.float64]): Motion capture data loaded with `np.loadtxt`,
            formatted with at least 18 columns of rotation data.

    Returns:
        Tuple[float, float]: A tuple of (left_rotation, right_rotation) in
            radians, representing total accumulated rotation for each arm.

    Raises:
        ValueError: If the input array does not have the expected shape or
            if motion capture data parsing fails.
    """
    try:
        left_rotation = calculate_total_rotation(data, 'L')
        right_rotation = calculate_total_rotation(data, 'R')

    except Exception as e:
        raise ValueError(f"Failed to parse motion capture data: {e}") from e

    return left_rotation, right_rotation
