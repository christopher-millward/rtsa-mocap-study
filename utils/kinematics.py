"""Functions for kinematics analysis.

Author: Christopher Millward
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt


def extract_rotation_matrix(row: npt.NDArray[np.float64], arm: str) -> npt.NDArray[np.float64]:
    """Extract a 3x3 rotation matrix from a 1D numeric row.

    The expected row layout is the 18-value array produced by `np.loadtxt`
    when reading the motion-capture file with the matrix columns only.

    Args:
        row (npt.NDArray[np.float64]): A 1D row of rotation data.
        arm (str): Arm identifier, either 'L' (left) or 'R' (right).

    Returns:
        npt.NDArray[np.float64]: A 3x3 rotation matrix with shape (3, 3).

    Raises:
        ValueError: If arm is not 'L' or 'R'.
        ValueError: If the row does not contain enough values.
    """
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    row_array = np.asarray(row, dtype=np.float64)
    if row_array.ndim != 1 or row_array.size < 18:
        raise ValueError(
            'Each row must be a 1D array with at least 18 numeric values.'
        )

    start_index = 0 if arm == 'L' else 9
    return row_array[start_index:start_index + 9].reshape(3, 3)


def calculate_rotation_angle(rotation_matrix: npt.NDArray[np.float64]) -> float:
    """Calculate the rotation angle in radians from a rotation matrix.

    Computes the angle of rotation from a 3x3 rotation matrix using the
    trace formula: θ = arccos((trace(R) - 1) / 2).

    Args:
        rotation_matrix (npt.NDArray[np.float64]): A 3x3 rotation matrix.

    Returns:
        float: Rotation angle in radians, value in range [0, π].

    Notes:
        - The result is always positive and in the range [0, π]
        - Due to numerical precision, trace values outside [-1, 3] are
          clamped to valid range before computing arccos

    Example:
        >>> R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Identity (no rotation)
        >>> angle = calculate_rotation_angle(R)
        >>> np.isclose(angle, 0)
        True
    """
    # Calculate trace (sum of diagonal elements)
    trace = np.trace(rotation_matrix)

    # Compute the argument for arccos
    # Clamp to [-1, 1] to handle numerical precision issues
    cos_angle = (trace - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate rotation angle
    angle = np.arccos(cos_angle)

    return float(angle)


def calculate_total_rotation(data: npt.NDArray[np.float64], arm: str) -> float:
    """Calculate total rotation magnitude for an arm across all frames.

    Args:
        data (npt.NDArray[np.float64]): A 2D array with 18 numeric columns per row.
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
    if data_array.ndim != 2 or data_array.shape[1] < 18:
        raise ValueError(
            'Motion capture data must be a 2D array with at least 18 columns.'
        )

    start_index = 0 if arm == 'L' else 9
    matrices = data_array[:, start_index:start_index + 9].reshape(-1, 3, 3)

    traces = np.trace(matrices, axis1=1, axis2=2)
    cos_angles = np.clip((traces - 1) / 2, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return float(angles.sum())


def calculate_arm_rotations(data: npt.NDArray[np.float64]) -> Tuple[float, float]:
    """Calculate total rotation for both arms from a motion capture array.

    Args:
        data (npt.NDArray[np.float64]): Motion capture data loaded with `np.loadtxt`.

    Returns:
        Tuple[float, float]: A tuple of (left_rotation, right_rotation) in
            radians, representing total accumulated rotation for each arm.

    Raises:
        ValueError: If the input array does not have the expected shape.
    """
    try:
        left_rotation = calculate_total_rotation(data, 'L')
        right_rotation = calculate_total_rotation(data, 'R')

    except Exception as e:
        raise ValueError(f"Failed to parse motion capture data: {e}") from e

    return left_rotation, right_rotation
