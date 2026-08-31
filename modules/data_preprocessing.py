"""Functions for cleaning and validating participant data before analysis.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt
from typing import TypedDict
from config import ORTHONORMAL_TOLERANCE, DETERMINANT_TOLERANCE
from modules.general_utilities import create_rotation_matrices

# Define the axis permutation and sign changes for aligning IMU axes with ISB axes
class ISBAxisTransform(TypedDict):
    columns: npt.NDArray[np.int32]
    signs: npt.NDArray[np.int32]

class ISBAxisTransforms(TypedDict):
    left: ISBAxisTransform
    right: ISBAxisTransform

_ISB_PERMUTATION: npt.NDArray[np.int32] = np.array([6, 7, 8, 3, 4, 5, 0, 1, 2])

_ISB_AXIS_TRANSFORMS: ISBAxisTransforms = {
    "left": {
        "columns": np.arange(0, 9),
        "signs": np.array([
            1, -1, -1, 
            -1, 1, 1, 
            1, -1, -1
        ]),
    },
    "right": {
        "columns": np.arange(9, 18),
        "signs": np.array([
            -1, -1, 1, 
            1, 1, -1, 
            -1, -1, 1
        ]),
    },
}


# ---- Functions ----
def align_axes_with_ISB(
    raw_data: npt.NDArray[np.float64],
    isb_permutation: npt.NDArray[np.int32] = _ISB_PERMUTATION,
    isb_axis_transforms: ISBAxisTransforms = _ISB_AXIS_TRANSFORMS,
) -> npt.NDArray[np.float64]:
    """Align flattened left and right arm rotation matrices with the ISB
    coordinate system.

    The input array contains one flattened 3x3 rotation matrix for each arm
    stored in a single array with shape ``(n_frames, 18)``:

        Columns 0-8
            Left arm

            [L00, L01, L02,
             L10, L11, L12,
             L20, L21, L22]

        Columns 9-17
            Right arm

            [R00, R01, R02,
             R10, R11, R12,
             R20, R21, R22]

    Each arm is transformed independently using a precomputed lookup table
    containing:

        - the columns corresponding to that arm,
        - the permutation of the nine matrix elements, and
        - the required sign changes.

    The lookup table is based on the following orientations of the IMUs
    relative to the ISB axes:

        | ISB Right Arm | Toroso IMU | Right IMU | Left IMU |
        | ------------- | ---------- | --------- | -------- |
        | X             | Z          | -X        | X        |
        | Y             | -Y         | -Y        | -Y       |
        | Z             | X          | Z         | -Z       |

    No intermediate 3x3 matrices are constructed. The transformation is
    performed entirely through vectorized indexing and multiplication.

    Args:
        raw_data:
            Array of flattened rotation matrices with shape ``(n_frames, 18)``.
        
        isb_permutation:
            Permutation applied to each flattened 3x3 rotation matrix to
            reorder its elements into the ISB coordinate system. The same
            permutation is used for both arms.

        isb_axis_transforms:
            Lookup table defining the ISB transformation for each arm. Each
            entry contains the column indices corresponding to the arm and
            the element-wise sign changes applied after the common
            permutation.

    Returns:
        Copy of ``raw_data`` with both arm rotation matrices aligned to the
        ISB coordinate system.

    Raises:
        ValueError:
            If the input is not numeric, has shape other than
            ``(n_frames, 18)``, is empty, or contains non-finite values.
    """

    try:
        data = np.asarray(raw_data, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "raw_data must contain only numeric values."
        ) from exc

    if data.ndim != 2 or data.shape[1] != 18:
        raise ValueError(
            "raw_data must have shape (n_frames, 18)."
        )

    if data.shape[0] == 0:
        raise ValueError(
            "raw_data must contain at least one frame."
        )

    if not np.isfinite(data).all():
        raise ValueError(
            "raw_data must contain only finite values."
        )

    for side in isb_axis_transforms.keys():
        transform = isb_axis_transforms[side]
        cols = transform["columns"]
        data[:, cols] = (
            data[:, cols][:, isb_permutation] * transform["signs"]
        )

    return data


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

    # ensure normalized matrix
    m = m / np.linalg.norm(m)
    target = target / np.linalg.norm(target)

    rot, _ = Rotation.align_vectors(m, target) #type: ignore
    return rot.as_matrix()


def apply_axis_orientation_correction(
    data: np.ndarray, 
    n_frames: int = 20
) -> np.ndarray:
    """
    Apply axis orientation correction to the given data.

    This function applies a correction matrix to the input data to align it with the specified target orientation.

    Args:
        data (np.ndarray): A 3D array of shape (n_frames, 3, 3) representing the rotation matrices for each frame.
        n_frames (int): The number of frames to use for calculating the average humerus direction.

    Returns:
        np.ndarray: The corrected data in the form of a 3D array of shape (n_frames, 3, 3) after applying the correction matrix.
    """
    # get the avg humerus direction for the first n_frames
    avg_hum_direction = data[:n_frames, :, :].mean(axis=0)

    # create correction matrix
    R_correction = get_correction_matrix(avg_hum_direction)

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
    Clean a batch of arm rotation matrices for analysis. Tasks:
        - Align the axes of the left and right arm rotation matrices with the ISB
          coordinate system.
        - Validate that the resulting matrices are proper rotation matrices.

    Args:
        raw_data:
            Array of flattened rotation matrices with shape ``(n_frames, 18)``.

    Returns:
        npt.NDArray[np.float64]: 
            Array of flattened rotation matrices with shape (``(n_frames, 18)``
            after cleaning and validation

    Raises:
        ValueError: If the input is not a batch of 3x3 matrices, if any matrix
            is not orthonormal, or if any determinant differs from ``+1``.
    """
    # Create rotation matrices from raw data
    data_right = create_rotation_matrices(raw_data, "right")
    data_left = create_rotation_matrices(raw_data, "left")

    # align axes with ISB CS
    clean_right = apply_axis_orientation_correction(data_right)
    clean_left = apply_axis_orientation_correction(data_left)

    # validate orthonormality and determinant
    validate_orthonorm_and_det(clean_right)
    validate_orthonorm_and_det(clean_left)

    # return cleaned data
    return cleaned_matrices
