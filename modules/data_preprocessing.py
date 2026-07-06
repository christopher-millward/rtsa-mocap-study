"""Functions for cleaning and validating participant data before analysis.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt
from typing import Literal


def align_axes_with_ISB(
    rotation_matrices: npt.NDArray[np.float64],
    arm: str,
) -> npt.NDArray[np.float64]:
    """Align a batch of arm rotation matrices with the ISB coordinate system.

    Applies the fixed change-of-basis transformation from the IMU axes to the
    ISB axes for the requested arm. The operation is vectorized across the
    entire batch of matrices.

    Args:
        rotation_matrices (npt.NDArray[np.float64]): Array of 3x3 rotation
            matrices with shape ``(n_frames, 3, 3)``.
        arm (str): Arm identifier, either ``'left'`` or ``'right'``.

    Returns:
        npt.NDArray[np.float64]: Batch of 3x3 matrices aligned to ISB axes.

    Raises:
        ValueError: If arm is not ``'left'`` or ``'right'``.
        ValueError: If the input is not a batch of 3x3 matrices, is empty,
            or contains non-finite values.
    """
    # Validate arm identifier.
    if arm not in ['left', 'right']:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")

    try:
        matrices = np.asarray(rotation_matrices, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rotation_matrices must contain only numeric values") from exc

    # Validate shape is (n_frames, 3, 3).
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotation_matrices must have shape (n_frames, 3, 3)")

    # Reject empty batches explicitly.
    if matrices.shape[0] == 0:
        raise ValueError("rotation_matrices must contain at least one matrix")

    if not np.isfinite(matrices).all():
        raise ValueError("rotation_matrices must contain only finite values")

    # These transformation matrices were determined from the IMUs being in the
    # following orientations relative to the ISB axes:
    #
    # | ISB Right Arm | Toroso IMU | Right IMU | Left IMU |
    # | ------------- | ---------- | --------- | -------- |
    # | X             | Z          | -X        | X        |
    # | Y             | -Y         | -Y        | -Y       |
    # | Z             | X          | Z         | -Z       |

    t_torso = np.array(
        [
            [0, 0, 1],
            [0, -1, 0],
            [1, 0, 0],
        ],
        dtype=np.float64,
    )
    t_right = np.array(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    t_left = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ],
        dtype=np.float64,
    )

    t_arm = t_left if arm == 'left' else t_right
    return t_torso @ matrices @ t_arm.T


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


def clean_and_validate_data(matrices: npt.NDArray[np.float64], arm: Literal['left', 'right']) -> npt.NDArray[np.float64]:
    """Clean a batch of arm rotation matrices for analysis.

    Args:
        matrices (npt.NDArray[np.float64]): Array of candidate rotation
            matrices with shape ``(n_steps, 3, 3)``.
        arm (Literal['left', 'right']): Arm identifier, either ``'left'`` or ``'right'``.

    Returns:
        npt.NDArray[np.float64]: An array of 3x3 rotation matrices with shape
            (n_frames, 3, 3).

    Raises:
        ValueError: If the input is not a batch of 3x3 matrices, if any matrix
            is not orthonormal, or if any determinant differs from ``+1``.
    """

    # Validate arm identifier
    if arm not in ['left', 'right']:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")

    # align axes for each arm
    cleaned_matrices = align_axes_with_ISB(matrices, arm)

    # validate orthonormality and determinant
    validate_orthonorm_and_det(cleaned_matrices)

    # return cleaned data
    return cleaned_matrices
