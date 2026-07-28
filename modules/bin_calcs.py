
"""Functions for calculating the amount of rotation about each axis.

Author: Christopher Millward
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Iterator, TypedDict
from scipy.spatial.transform import Rotation as R
from schema import Heatmap
from modules.general_utilities import create_rotation_matrices
from modules.data_preprocessing import validate_orthonorm_and_det

# ----------------------------
# Local-only classes
# ----------------------------
@dataclass
class PosturalAngles:
    """Anatomical posture angles in degrees."""
    poe: npt.NDArray[np.float64]
    elevation: npt.NDArray[np.float64]
    ir_er: npt.NDArray[np.float64]

@dataclass
class BinBounds:
    elevation_start: int
    elevation_end: int
    poe_start: int
    poe_end: int

@dataclass(slots=True)
class BinRotationResult:
    elevation: np.float64
    poe: np.float64
    ir_er: np.float64
    cumulative_motion: np.float64
    sample_count: int


# ----------------------------
# Functions
# ----------------------------

def _validate_rotation_data(
    data: npt.NDArray[np.float64],
    arm: Literal["left", "right"],
) -> npt.NDArray[np.float64]:
    """
    Validate incoming data and convert it to np.float64 dtype.

    Args:
        data (npt.NDArray[np.float64]): Flattened rotation matrices with shape (n_frames, 18).
        arm (Literal["left", "right"]): Arm to process.

    Returns:
        npt.NDArray[np.float64]: Validated numpy array of shape (n_frames, 18).

    Raises:
        ValueError: If inputs are invalid.
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

    return data_array


def _get_postural_angles(
    rotation_matrices: npt.NDArray[np.float64],
) -> PosturalAngles:
    """Calculate the postural angles, in degrees, (POE, elevation, IR/ER) from
    rotation matrices for each frame.

    Args:
        rotation_matrices:
            Array of rotation matrices with shape (n_frames, 3, 3).

    Returns:
        PosturalAngles object containing POE, elevation, and IR/ER angles in degrees, with shape (n_frames,).

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

    return PosturalAngles(
        poe=euler_angles[:, 0],
        elevation=euler_angles[:, 1],
        ir_er=euler_angles[:, 2],
    )


def _normalize_postural_angles(
    angles: PosturalAngles,
) -> PosturalAngles:
    """Normalize Euler postural angles for workspace binning.

    POE and IR_ER angles are normalized to the range [0, 360) degrees.
    Elevation angles are normalized to the range [0, 180] degrees.

    Args:
        angles (PosturalAngles): Postural angles in degrees.

    Returns:
        PosturalAngles: Normalized postural angles in degrees.
    """
    if not isinstance(angles, PosturalAngles):
        raise TypeError(
            "Input must be a PosturalAngles instance."
        )

    # Make sure fields are not empty
    if angles.poe is None or angles.elevation is None or angles.ir_er is None:
        raise ValueError(
            "PosturalAngles fields cannot be None."
        )

    return PosturalAngles(
        poe=np.mod(angles.poe, 360.0),
        elevation=np.mod(np.abs(angles.elevation), 180.0),
        ir_er=np.mod(angles.ir_er, 360.0),
    )


def _compute_incremental_rotation_matrices(
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
    # keep batch axis fixed, swap matrix axes
    deltas = np.matmul(curr, np.transpose(prev, (0, 2, 1)))

    return deltas


def _calculate_relative_motion(
    data: npt.NDArray[np.float64],
    arm: Literal["left", "right"],
) -> tuple[
    npt.NDArray[np.float64],
    PosturalAngles,
]:
    """
    Calculate relative rotations and postural angles.

    Args:
        data (npt.NDArray[np.float64]): Flattened rotation matrices with shape (n_frames, 18).
        arm (Literal["left", "right"]): Arm to process.

    Returns:
        tuple[npt.NDArray[np.float64], PosturalAngles]: A tuple containing:
            - Relative rotation matrices
            - Normalized postural angles
    """
    matrices = create_rotation_matrices(data, arm)

    postural_angles = _get_postural_angles(matrices)
    postural_angles = _normalize_postural_angles(postural_angles)

    relative_matrices = _compute_incremental_rotation_matrices(
        matrices
    )

    return relative_matrices, postural_angles


def _generate_heatmap_bins(
    bin_width: int,
    elevation_range_end: int,
    poe_range_end: int,
) -> Iterator[BinBounds]:
    """
    Generate elevation and POE bin boundaries.

    Args:
        bin_width (int): The width of each bin.
        elevation_range_end (int): The end of the elevation range.
        poe_range_end (int): The end of the POE range.

    Yields:
        Iterator[BinBounds]: Iterator holding a datacclass containing:
            elevation_start,
            elevation_end,
            poe_start,
            poe_end
    """
    for elevation_start in range(0, elevation_range_end, bin_width):
        for poe_start in range(0, poe_range_end, bin_width):
            yield BinBounds(
                elevation_start=elevation_start,
                elevation_end=elevation_start + bin_width,
                poe_start=poe_start,
                poe_end=poe_start + bin_width
            )


def _extract_bin_data(
    mocap_data: npt.NDArray[np.float64],
    postural_data: PosturalAngles,
    elevation_start: float,
    elevation_end: float,
    poe_start: float,
    poe_end: float,
) -> npt.NDArray[np.float64]:
    """Extract rows of data that fall within a specified elevation and POE bin.

    Args:
        mocap_data (npt.NDArray[np.float64]): A series of R matrices with shape (n_frames, 3, 3) representing the relative rotation matrices for each frame.

        postural_data (PosturalAngles): The postural angles for each frame in shape (n_frames,). Expected columns:
            0 = POE, 
            1 = Elevation, 
            2 = IR/ER.

        elevation_start (float): Lower elevation bound (inclusive).
        elevation_end (float): Upper elevation bound (exclusive).

        poe_start (float): Lower POE bound (inclusive).
        poe_end (float): Upper POE bound (exclusive).

    Returns:
        npt.NDArray[np.float64]: Subset of the original data that falls within
        the specified elevation and POE bin.

    Raises:
        ValueError: If either bin has invalid bounds.
        IndexError: If the input array does not contain enough columns.
    """

    # Validate input dimensions
    # mocap data shape n, 3, 3
    if mocap_data.shape[1:] != (3, 3):
        raise ValueError("mocap_data must have shape (n_frames, 3, 3)")

    # postural data needs 3 cols
    if not isinstance(postural_data, PosturalAngles):
        raise TypeError("postural_data must be a PosturalAngles instance")

    # postural data cannot be none
    if postural_data.poe is None or postural_data.elevation is None:
        raise ValueError(
            "postural_data must have non-None poe and elevation arrays")

    # Validate bin widths
    if elevation_start >= elevation_end:
        raise ValueError(
            f"elevation_start {elevation_start} must be less than "
            f"elevation_end {elevation_end}."
        )

    if poe_start >= poe_end:
        raise ValueError(
            f"poe_start {poe_start} must be less than poe_end {poe_end}."
        )

    # Create masks for both dimensions
    # Since we're applying the masks to the relative rotation matrices,
    # they need to have length n_frames - 1, which is why the last row
    # is omitted from each.
    elevation_mask = (
        (postural_data.elevation >= elevation_start)
        & (postural_data.elevation < elevation_end)
    )[:-1]

    poe_mask = (
        (postural_data.poe >= poe_start)
        & (postural_data.poe < poe_end)
    )[:-1]

    # Keep only rows satisfying both conditions
    return mocap_data[elevation_mask & poe_mask]


def _decompose_rotation_matrices_yxy(
    relative_rotations: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Decompose relative rotations into absolute Euler components using Y-X-Y sequence.

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
        npt.NDArray[np.float64]: Absolute euler angles in radians with shape
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


def _accumulate_euler_components(
    euler_angles: npt.NDArray[np.float64],
) -> Tuple[np.float64, np.float64, np.float64]:
    """Sum Euler components independently across all timestep transitions.

    Given a matrix of decomposed Euler angles, this function will first take 
    the absolute value of each angle to ensure all values are positive, then
    sum each component independently to produce cumulative motion values for 
    the three sequence positions. 

    Args:
        euler_angles (npt.NDArray[np.float64]): Array of Euler angles with shape
            (n_steps, 3).

    Returns:
        Tuple[np.float64, np.float64, np.float64]: Cumulative sums of the first,
        second, and third Euler components, respectively.

    Raises:
        ValueError: If input does not have shape (n_steps, 3).
        ValueError: If input is empty.
    """

    # Ensure correct dtype
    all_components = np.asarray(euler_angles, dtype=np.float64)

    # Validate shape is 2D with exactly 3 columns
    if all_components.ndim != 2 or all_components.shape[1] != 3:
        raise ValueError("euler_angles must have shape (n_steps, 3)")

    # Reject empty batch
    if all_components.shape[0] == 0:
        raise ValueError("euler_angles must contain at least one row")

    # Coerce all values to be non-negative
    all_components = np.abs(all_components)

    # Sum each column (component) across rows and convert to native floats
    sums: npt.NDArray[np.float64] = np.sum(all_components, axis=0)

    return (sums[0], sums[1], sums[2])


def _calculate_single_bin(
    relative_matrices: npt.NDArray[np.float64],
    postural_angles: PosturalAngles,
    elevation_start: int,
    elevation_end: int,
    poe_start: int,
    poe_end: int,
) -> BinRotationResult:
    """Calculate cumulative motion metrics for one heatmap bin.

    Args:
        relative_matrices (npt.NDArray[np.float64]): Relative rotation matrices with shape (n_frames - 1, 3, 3).
        postural_angles (PosturalAngles): Normalized postural angles for each frame.
        elevation_start (int): Lower elevation bound (inclusive).
        elevation_end (int): Upper elevation bound (exclusive).
        poe_start (int): Lower POE bound (inclusive).
        poe_end (int): Upper POE bound (exclusive).


    Returns:
        BinRotationResult: A data class containing the calculated metrics.
    """
    # Extract data from current bin
    bin_data = _extract_bin_data(
        mocap_data=relative_matrices,
        postural_data=postural_angles,
        elevation_start=elevation_start,
        elevation_end=elevation_end,
        poe_start=poe_start,
        poe_end=poe_end,
    )

    # return zero values if no data in the bin
    if bin_data.shape[0] == 0:
        return BinRotationResult(
            elevation=np.float64(0),
            poe=np.float64(0),
            ir_er=np.float64(0),
            cumulative_motion=np.float64(0),
            sample_count=0
        )
    
    # decompose data into euler angles
    euler_angles = _decompose_rotation_matrices_yxy(bin_data)

    # sum rotations
    elevation, poe, ir_er = (_accumulate_euler_components(euler_angles))

    # get total motion and sample count
    total_motion = elevation + poe + ir_er
    n_samples = bin_data.shape[0]

    return BinRotationResult(
        elevation=elevation,
        poe=poe,
        ir_er=ir_er,
        cumulative_motion=total_motion,
        sample_count=n_samples,
    )


def _add_bin_result_to_heatmap(
    heatmap: Heatmap,
    result: BinRotationResult,
) -> None:
    """Helper function to append a single bin result to a heatmap.

    Args:
        heatmap (Heatmap): The heatmap to update.
        result (BinRotationResult): The result to append.

    Returns:
        None: The heatmap is updated in place.
    """

    heatmap.elevation = np.append(heatmap.elevation, result.elevation)
    heatmap.poe = np.append(heatmap.poe, result.poe)
    heatmap.ir_er = np.append(heatmap.ir_er, result.ir_er)
    heatmap.cumulative_motion = np.append(heatmap.cumulative_motion, result.cumulative_motion)
    heatmap.sample_count = np.append(heatmap.sample_count, result.sample_count)


def _populate_heatmap(
    relative_matrices: npt.NDArray[np.float64],
    postural_angles: PosturalAngles,
    heatmap: Heatmap,
) -> Heatmap:
    """Populate heatmap with rotational motion data.

    Args:
        relative_matrices (npt.NDArray[np.float64]): Relative rotation matrices with shape (n_frames - 1, 3, 3).
        postural_angles (PosturalAngles): Normalized postural angles for each frame.
        heatmap (Heatmap): Heatmap object to populate.

    Returns:
        Heatmap: Updated heatmap with calculated metrics for each bin.
    """

    for bin_bounds in _generate_heatmap_bins(
        heatmap.bin_width,
        heatmap.elevation_range_end,
        heatmap.poe_range_end
    ):
        result = _calculate_single_bin(
            relative_matrices,
            postural_angles,
            **bin_bounds.__dict__,
        )

        _add_bin_result_to_heatmap(
            heatmap,
            result,
        )

    return heatmap


def calculate_bin_rotations(
    data: npt.NDArray[np.float64],
    arm: Literal['left', 'right']
):

    # validate incoming data
    data_array = _validate_rotation_data(data, arm)

    # calculate relative motion
    relative_matrices, postural_angles = _calculate_relative_motion(data_array, arm)

    # Initialize heatmap object
    heatmap = Heatmap()

    # return populated heatmap
    return _populate_heatmap(relative_matrices, postural_angles, heatmap)
