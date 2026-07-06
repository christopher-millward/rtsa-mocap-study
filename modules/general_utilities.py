"""Functions for helper functions used in the kinematics pipeline.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt


def create_rotation_matrices(
    data: npt.NDArray[np.float64],
    arm: str,
) -> npt.NDArray[np.float64]:
    """Extract a batch of 3x3 rotation matrices for a specified arm.

    The function operates on a 2D motion-capture array and slices the nine
    rotation values for the requested arm from every frame. Those nine values
    are reshaped into a 3x3 matrix for each frame, producing a vectorized
    stack of rotation matrices.

    Args:
        data (npt.NDArray[np.float64]): A 2D array with exactly 18 columns,
            where columns 0-8 contain left arm rotation data and columns
            9-17 contain right arm rotation data.
        arm (str): Arm identifier, either 'left' or 'right'.

    Returns:
        npt.NDArray[np.float64]: An array of 3x3 rotation matrices with shape
            (n_frames, 3, 3).

    Raises:
        ValueError: If arm is not 'left' or 'right'.
        ValueError: If the row does not contain enough values.
    """
    # Validate arm identifier
    if arm not in ['left', 'right']:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")

    # validate data shape
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError('Data must be a 2D array with exactly 18 columns.')

    # reject empty data
    if data_array.shape[0] == 0:
        raise ValueError("Data must be a 2D array with exactly 18 columns.")

    start_index = 0 if arm == 'left' else 9
    return data_array[:, start_index:start_index + 9].reshape(-1, 3, 3)
