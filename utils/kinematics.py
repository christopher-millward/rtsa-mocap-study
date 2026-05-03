"""Kinematics analysis functions for rotation data processing.

Author: Christopher Millward
"""

from typing import Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd


def extract_rotation_matrix(row: pd.Series, arm: str) -> npt.NDArray[np.float64]:
    """Extract a 3x3 rotation matrix from a data row.

    Extracts the nine rotation matrix elements for the specified arm from
    a row of motion capture data and returns them as a 3x3 NumPy array.

    Args:
        row (pd.Series): A row from the motion capture data DataFrame.
        arm (str): Arm identifier, either 'L' (left) or 'R' (right).

    Returns:
        npt.NDArray[np.float64]: A 3x3 rotation matrix with shape (3, 3).

    Raises:
        ValueError: If arm is not 'L' or 'R'.
        KeyError: If required matrix columns are missing from the row.

    Example:
        >>> row_data = pd.Series({'L00': 0.1, 'L01': -0.37, ..., 'L22': 0.41})
        >>> R = extract_rotation_matrix(row_data, 'L')
        >>> R.shape
        (3, 3)
    """
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    # Column names for the rotation matrix elements
    columns = [f'{arm}{i}{j}' for i in range(3) for j in range(3)]

    try:
        values: npt.NDArray[np.float64] = np.array(
            row[columns].values, dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"Missing rotation matrix columns for arm {arm}: {e}")

    # Reshape into 3x3 matrix
    return values.reshape(3, 3)


def calculate_rotation_angle(rotation_matrix: npt.NDArray[np.float64]) -> float:
    """Calculate the rotation angle in radians from a rotation matrix.

    Computes the angle of rotation from a 3x3 rotation matrix using theeyr
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


def calculate_total_rotation(dataframe: pd.DataFrame, arm: str) -> float:
    """Calculate total rotation magnitude for an arm across all frames.

    Processes all rows in a motion capture DataFrame and computes the
    accumulated rotation angle for the specified arm.

    Args:
        dataframe (pd.DataFrame): Motion capture data with rotation matrix columns.
        arm (str): Arm identifier, either 'L' (left) or 'R' (right).

    Returns:
        float: Total accumulated rotation in radians.

    Raises:
        ValueError: If arm is not 'L' or 'R'.
        KeyError: If required rotation matrix columns are missing.

    Example:
        >>> df = pd.read_csv('motion_data.csv')
        >>> left_rotation = calculate_total_rotation(df, 'L')
        >>> right_rotation = calculate_total_rotation(df, 'R')
    """
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    total_rotation = 0.0

    for _, row in dataframe.iterrows():
        rotation_matrix = extract_rotation_matrix(row, arm)
        angle = calculate_rotation_angle(rotation_matrix)
        total_rotation += angle

    return total_rotation


def calculate_arm_rotations(filepath: str) -> Tuple[float, float]:
    """Calculate total rotation for both arms from a motion capture file.

    Loads motion capture data from a file and computes the total accumulated
    rotation angle for both the left and right arms.

    Args:
        filepath (str): Path to a motion capture data file (e.g., TSV, CSV).

    Returns:
        Tuple[float, float]: A tuple of (left_rotation, right_rotation) in
            radians, representing total accumulated rotation for each arm.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required rotation matrix columns are missing.

    Example:
        >>> left_rot, right_rot = calculate_arm_rotations('./data/1_R_MATRICES 4-8-2015')
        >>> print(f"Left arm: {left_rot:.2f} rad, Right arm: {right_rot:.2f} rad")
        Left arm: 45.23 rad, Right arm: 38.91 rad
    """
    # Try to load the file as TSV (with tab separator)
    try:
        df = pd.read_csv(filepath, sep='\t')
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Motion capture file not found: {filepath}") from e
    except Exception as e:
        raise ValueError(f"Failed to parse motion capture file: {e}") from e

    # Calculate total rotation for each arm
    left_rotation = calculate_total_rotation(df, 'L')
    right_rotation = calculate_total_rotation(df, 'R')

    return left_rotation, right_rotation
