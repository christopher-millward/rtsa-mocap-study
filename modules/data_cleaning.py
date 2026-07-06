"""Functions for cleaning participant data before analysis.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt


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
        raise ValueError("rotation_matrices must contain only numeric values") from exc

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



def clean_data():
    # align axes for each arm
    # validate orthonormality and determinant
    # return cleaned data
    pass