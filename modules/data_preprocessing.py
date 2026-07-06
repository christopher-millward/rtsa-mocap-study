"""Functions for cleaning and validating participant data before analysis.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt
from typing import Literal
from config import ORTHONORMAL_TOLERANCE, DETERMINANT_TOLERANCE

# Flattened 3x3 rotation matrix indexing:
#
#     [00, 01, 02,
#      10, 11, 12,
#      20, 21, 22]
#
# corresponds to:
#
#     [[00, 01, 02],
#      [10, 11, 12],
#      [20, 21, 22]]
#
# The transformation:
#
#     R_ISB = T_torso @ R_IMU @ T_arm.T
#
# is implemented as a fixed column permutation and sign flip because
# all transformation matrices contain only axis swaps and sign changes.
#
# The values represent:
#
#     output = input[:, indices] * signs
#
_ISB_AXIS_TRANSFORMS = {
    "right": (
        np.array([6, 7, 8, 3, 4, 5, 0, 1, 2]),
        np.array([-1, -1, 1, 1, 1, -1, -1, -1, 1]),
    ),
    "left": (
        np.array([6, 7, 8, 3, 4, 5, 0, 1, 2]),
        np.array([1, 1, -1, -1, -1, 1, 1, 1, -1]),
    ),
}



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

def align_axes_with_ISB_flat(
    raw_data: npt.NDArray[np.float64],
    arm: str,
) -> npt.NDArray[np.float64]:
    """Align flattened arm rotation matrices with the ISB coordinate system.

    Applies the fixed change-of-basis transformation from the IMU coordinate
    system to the International Society of Biomechanics (ISB) humerothoracic 
    coordinate system for the right shoulder.

    The input contains flattened rotation matrices for both arms stored in a
    single array:

        Columns 0-8:
            Left arm rotation matrix:

            [L00, L01, L02,
             L10, L11, L12,
             L20, L21, L22]

        Columns 9-17:
            Right arm rotation matrix:

            [R00, R01, R02,
             R10, R11, R12,
             R20, R21, R22]

    The function does not construct intermediate 3x3 rotation matrices.
    Instead, it performs the equivalent operation:

        R_ISB = T_torso @ R_IMU @ T_arm.T

    through direct indexing and sign changes.

    For example, the right arm transformation converts:

        [R00, R01, R02,
         R10, R11, R12,
         R20, R21, R22]

    into:

        [-R20, -R21,  R22,
          R10,  R11, -R12,
         -R00, -R01,  R02]

    The lookup table stores these index selections and sign changes to
    efficiently transform every frame in a vectorized operation.

    Args:
        raw_data (npt.NDArray[np.float64]):
            Array containing both flattened rotation matrices with shape ``(n_frames, 18)``.

        arm (str):
            Arm identifier specifying which rotation matrix to transform.
            Must be either ``'left'`` or ``'right'``.

    Returns:
        npt.NDArray[np.float64]:
            Copy of the input array with the selected arm rotation matrix
            aligned to the ISB coordinate system. All non-selected columns
            remain unchanged.

    Raises:
        ValueError:
            If ``arm`` is not ``'left'`` or ``'right'``.

        ValueError:
            If the input does not have shape ``(n_frames, 19)``, is empty,
            or contains non-finite values.

    Notes:
        Axis mappings used to derive the transformation:

        ```
        | ISB Axis | Right IMU | Left IMU |
        |----------|-----------|----------|
        | X        | -X        | X        |
        | Y        | -Y        | -Y       |
        | Z        | Z         | -Z       |
        ```

        The transformation is performed in-place on a copied array to avoid
        modifying the original ingestion data.
    """

    if arm not in _ISB_AXIS_TRANSFORMS:
        raise ValueError(
            f"arm must be 'left' or 'right', got {arm!r}"
        )

    # Validate arm identifier.
    if arm not in ['left', 'right']:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")

    try:
        data = np.asarray(raw_data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "raw_data must contain only numeric values") from exc

    # Validate shape is (n_frames, 3, 3).
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("raw_data must have shape (n_frames, 3, 3)")

    # Reject empty batches explicitly.
    if data.shape[0] == 0:
        raise ValueError("raw_data must contain at least one matrix")

    if not np.isfinite(data).all():
        raise ValueError("raw_data must contain only finite values")

    # Select the appropriate column range for the specified arm. 
    start = 0 if arm == "left" else 9
    stop = start + 9
    # Lookup the index permutation and sign changes for the specified arm.
    indices, signs = _ISB_AXIS_TRANSFORMS[arm]
    # Apply the transformation
    data[:, start:stop] = (
        data[:, start:stop][:, indices] * signs
    )

    return data


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
    cleaned_matrices = align_axes_with_ISB_flat(matrices, arm)

    # validate orthonormality and determinant
    validate_orthonorm_and_det(cleaned_matrices)

    # return cleaned data
    return cleaned_matrices
