"""Functions for cleaning and validating participant data before analysis.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation
from config import ORTHONORMAL_TOLERANCE, DETERMINANT_TOLERANCE

# ---- Functions ----
def get_correction_matrix(
    m: np.ndarray,
    target: np.ndarray = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
):
    """
    Return a rotation matrix that rotates a matrix to align with the specified target matrix (default is the origin axes).

    Args:
        m (np.ndarray): A 3D matrix to be rotated.
        target (np.ndarray): The target matrix to align with (default is the origin axes).

    Returns:
        np.ndarray: A 3x3 rotation matrix that rotates m to align with the target matrix.
    """
    # ensure shape is (3, 3)
    if m.shape != (3, 3) or target.shape != (3, 3):
        raise ValueError("m and target must have shape (3, 3)")

    # ensure normalized matrix
    m = m / np.linalg.norm(m)
    target = target / np.linalg.norm(target)

    rot, _ = Rotation.align_vectors(m, target) #type: ignore
    return rot.as_matrix()


def apply_axis_orientation_correction(
    data: np.ndarray, 
    n_frames: int = 20,
    target: np.ndarray = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
) -> np.ndarray:
    """
    Apply axis orientation correction to the given data.

    This function applies a correction matrix to the input data to align it with the specified target orientation.

    Args:
        data (np.ndarray): A 3D array of shape (n_frames, 3, 3) representing the rotation matrices for each frame.
        n_frames (int): The number of frames to use for calculating the average humerus direction.
        target (np.ndarray): The target orientation to align the data with.

    Returns:
        np.ndarray: The corrected data in the form of a 3D array of shape (n_frames, 3, 3) after applying the correction matrix.
    """
    # ensure shape is (n_frames, 3, 3)
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("data must have shape (n_frames, 3, 3)")

    # Ensure data is not empty
    if data.shape[0] == 0:
        raise ValueError("data must contain at least one frame")
    
    # ensure n_frames is not greater than the number of frames in data
    if n_frames > data.shape[0]:    
        raise ValueError("n_frames must not be greater than the number of frames in data")  

    # get the avg humerus direction for the first n_frames
    avg_hum_direction = data[:n_frames, :, :].mean(axis=0)

    # create correction matrix
    R_correction = get_correction_matrix(avg_hum_direction, target)

    # Apply correction to every frame
    return R_correction @ data  # NOTE: This order matters!! Don't rearrange!!


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

    # Calculate errors
    gram = np.matmul(np.transpose(data_array, (0, 2, 1)), data_array)
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)
    orthonormal_error = np.abs(gram - identity)
    max_orthonormal_error = np.max(orthonormal_error)

    if max_orthonormal_error > ORTHONORMAL_TOLERANCE:
        raise ValueError(
            f"matrices must be orthonormal rotation matrices. "
            f"Largest orthonormality error: {max_orthonormal_error:.3e}"
        )

    dets = np.linalg.det(data_array)
    det_errors = np.abs(dets - 1.0)
    max_det_error = np.max(det_errors)

    if max_det_error > DETERMINANT_TOLERANCE:
        raise ValueError(
            f"matrices must have a determinant of 1. "
            f"Largest determinant error: {max_det_error:.3e}"
        )


def clean_and_validate_data(raw_data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Clean a batch of arm rotation matrices for analysis. 
    Tasks:
        - Perform axis alignment to the ISB coordinate system.
        - Validate that the resulting matrices are proper rotation matrices.

    Args:
        raw_data:
            Array of rotation matrices with shape ``(n_frames, 3, 3)``.

    Returns:
        npt.NDArray[np.float64]: 
            Array of rotation matrices with shape (``(n_frames, 3, 3)``
            after cleaning and validation.
    """
    # coerce to float64
    data = np.asarray(raw_data, dtype=np.float64)

    # align axes with ISB CS
    clean_data = apply_axis_orientation_correction(data)

    # validate orthonormality and determinant
    validate_orthonorm_and_det(clean_data)

    # return cleaned data
    return clean_data
