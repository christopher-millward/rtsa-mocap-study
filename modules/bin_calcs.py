
"""Functions for calculating the amount of rotation about each axis.

Author: Christopher Millward
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.spatial.transform import Rotation as R
from modules.general_utilities import create_rotation_matrices
from modules.data_preprocessing import validate_orthonorm_and_det

@dataclass
class PositionAngles():
    """Absolute anatomical position angles in degrees."""

    poe: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64]
    ir_er: npt.NDArray[np.float64]


def get_position_angles(
    rotation_matrices: npt.NDArray[np.float64],
) -> PositionAngles:
    """Calculate the postural angles, in degrees, (POE, elevation, IR/ER) from
    rotation matrices for each frame.

    Args:
        rotation_matrices:
            Array of rotation matrices with shape (n_frames, 3, 3).

    Returns:
        PositionAngles object containing POE, elevation, and IR/ER angles in degrees, with shape (n_frames,).

    Raises:
        ValueError:
            If the input does not have shape (n_frames, 3, 3).
    """

    if rotation_matrices.ndim != 3:
        raise ValueError(
            "rotation_matrices must have shape (n_frames, 3, 3)."
        )

    if rotation_matrices.shape[1:] != (3, 3):
        raise ValueError(
            "rotation_matrices must have shape (n_frames, 3, 3)."
        )

    rotations = R.from_matrix(rotation_matrices)

    # ISB shoulder convention
    # Intrinsic Y-X-Y decomposition
    euler_angles = rotations.as_euler(
        "YXY",
        degrees=True,
    )

    return PositionAngles(
        poe=euler_angles[:, 0],
        elevation=euler_angles[:, 1],
        ir_er=euler_angles[:, 2],
    )

def normalize_position_angles(
    angles: PositionAngles,
) -> PositionAngles:
    """Normalize Euler position angles for workspace binning.
    
    POE and IR_ER angles are normalized to the range [0, 360) degrees.
    Elevation angles are normalized to the range [0, 180] degrees.
    """

    return PositionAngles(
        poe=np.mod(angles.poe, 360.0),
        elevation=np.mod(np.abs(angles.elevation), 180.0),
        ir_er=np.mod(angles.ir_er, 360.0),
    ) 

def compute_incremental_rotation_matrices(
    rotation_matrices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build per-timestep relative rotation matrices from absolute orientations.

    For each timestep t > 0, this function will compute the relative rotation
    that maps the previous frame orientation to the current frame orientation.
    The intended vectorized operation is:
        R_delta[t] = R_current[t] @ R_previous[t].T

    Args:
        rotation_matrices (npt.NDArray[np.float64]): Absolute rotation matrices
            for one arm with shape (n_frames, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Relative rotation matrices for each transition
            with shape (n_frames - 1, 3, 3).

    Raises:
        ValueError: If input does not have shape (n_frames, 3, 3), has fewer
            than two frames, is not orthonormal, or has determinants that differ from 1.
    """
    
    # Coerce to ndarray with correct dtype
    matrices = np.asarray(rotation_matrices, dtype=np.float64)

    # Validate shape is (n_frames, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(
            'rotation_matrices must have shape (n_frames, 3, 3)'
        )

    # Validate shape has at least 2 frames
    n_frames = matrices.shape[0]
    if n_frames < 2:
        raise ValueError(
            'rotation_matrices must contain at least two frames to compute increments'
        )

    # Previous and current frames (vectorized stacks)
    prev = matrices[:-1]
    curr = matrices[1:]

    # Compute relative rotations: R_delta[t] = R_current[t] @ R_previous[t].T
    deltas = np.matmul(curr, np.transpose(prev, (0, 2, 1))) # keep batch axis fixed, swap matrix axes 

    return deltas

def decompose_rotation_matrices_yxy(
    relative_rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Decompose relative rotations into Euler components using Y-X-Y sequence.

    This function will perform a batch decomposition of each 3x3 relative
    rotation matrix into three Euler angles following the ISB-recommended 
    Y-X-Y convention, returning one angle triplet per timestep transition.

    Using the Y-X-Y convention allows us to map the following rotations to 
    the anatomical axes of the arm (as per G. Wu et al (2005), Fig. 7):
        - First Y: plane of elevation
        - X: elevation
        - Second Y: axial rotation

    The lowercase 'yxy' sequence is used for intrinsic rotations, which allows
    us to interpret the angles as rotations about the moving axes of the arm.
    
    Args:
        relative_rotations (npt.NDArray[np.float64]): Relative rotation matrices
            with shape (n_steps, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Euler angles in radians with shape
            (n_steps, 3), ordered as (first_Y, X, second_Y).

    Raises:
        ValueError: If the input shape is invalid.
    """
    # Coerce to ndarray with correct dtype
    matrices = np.asarray(relative_rotations, dtype=np.float64)


    # Validate shape is (n_frames, 3, 3)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("relative_rotations must have shape (n_steps, 3, 3)")
    
    # Reject empty batch explicitly
    if matrices.shape[0] == 0:
        raise ValueError("relative_rotations must contain at least one matrix")

    # Validate orthonormality and determinant of each matrix
    validate_orthonorm_and_det(matrices)

    euler_angles = R.from_matrix(matrices).as_euler(
        seq="YXY",  # Must be uppercase for intrinsic rotations in scipy
        degrees=False,
    )

    return np.asarray(euler_angles, dtype=np.float64)




def calculate_bin_rotations(
        data: npt.NDArray[np.float64], 
        arm: Literal['left', 'right']
    ):
    """
    Args:
        data: Array of flattened rotation matrices with shape (n_frames, 18).
        arm: The arm for which to calculate bin rotations.
    """
    # Validate arm
    if arm not in ['left', 'right']:
        raise ValueError(f"arm must be 'left' or 'right', got {arm}")

    # Validate data shape
    data_array = np.asarray(data, dtype=np.float64)
    if data_array.ndim != 2 or data_array.shape[1] != 18:
        raise ValueError(
            'Data must be a 2D array with exactly 18 columns.'
        )

    # Validate data is not empty
    if data_array.shape[0] == 0:
        raise ValueError("Input data cannot be empty")

    # Create absolute rotation matrices for the specified arm
    matrices = create_rotation_matrices(data_array, arm)

    # determine absolute position angles (elevation, poe, ir_er) for each frame
    absolute_angles = get_position_angles(matrices)
    absolute_angles = normalize_position_angles(absolute_angles)

    # compute relative motion between each frame

    # decompose relative motion matrices into euler angles

    # for each bin:
        # extract bin boundaries
        # filter extract only data within the bin boundaries
            # create bin mask based ont he absolute position angles
                # base it on start position
            # use the mask to return only the data within the bin boundaries
        # sum motion in each axis for the bin
        # save the bin calcs to the data object
        





    raise NotImplementedError("This function is not yet implemented.")