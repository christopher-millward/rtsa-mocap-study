"""Functions for helper functions used in the kinematics pipeline.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt


def validate_orthonorm_and_det(matrices: npt.NDArray[np.float64]) -> None:
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
    # Validate shape is (n_frames, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("matrices must have shape (n_frames, 3, 3)")
    
    # Reject empty batch explicitly
    if matrices.shape[0] == 0:
        raise ValueError("batch must contain at least one matrix")
    
    # Coerce to float64
    data_array = np.asarray(matrices, dtype=np.float64)

    gram = np.matmul(np.transpose(data_array, (0, 2, 1)), data_array)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)
    if not np.allclose(gram, identity, atol=1e-8):
        raise ValueError("matrices must be orthonormal rotation matrices")

    dets = np.linalg.det(data_array)
    if not np.allclose(dets, 1.0, atol=1e-6):
        raise ValueError("matrices must have a determinant of 1")


def create_rotation_matrices(data: npt.NDArray[np.float64], arm: str) -> npt.NDArray[np.float64]:
    """Extract a batch of 3x3 rotation matrices for a specified arm.

    The function operates on a 2D motion-capture array and slices the nine
    rotation values for the requested arm from every frame. Those nine values
    are reshaped into a 3x3 matrix for each frame, producing a vectorized
    stack of rotation matrices.

    Args:
        data (npt.NDArray[np.float64]): A 2D array with exactly 18 columns,
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
    # Validate arm identifier
    if arm not in ['L', 'R']:
        raise ValueError(f"arm must be 'L' or 'R', got {arm}")

    # validate data shape
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError(
            'Data must be a 2D array with exactly 18 columns.'
        )

    # reject empty data
    if data_array.shape[0] == 0:
        raise ValueError("Data must be a 2D array with exactly 18 columns.")

    start_index = 0 if arm == 'L' else 9
    return data_array[:, start_index:start_index + 9].reshape(-1, 3, 3)

