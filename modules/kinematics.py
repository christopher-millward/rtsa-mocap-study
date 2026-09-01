
"""Functions for calculating the amount of rotation about each axis.

Author: Christopher Millward
"""

from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Iterator
from scipy.spatial.transform import Rotation as R
from schema import Heatmap
from modules.general_utilities import create_rotation_matrices
from modules.data_preprocessing import validate_orthonorm_and_det
from modules.progress_bar import get_pbar_manager

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
    data: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Validate incoming data and ensure it is of type np.float64.

    Args:
        data (npt.NDArray[np.float64]): Array of rotation matrices with shape (n_frames, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Validated numpy array of shape (n_frames, 18).

    Raises:
        ValueError: If inputs are invalid.
    """
    # Validate data shape is (n_frames, 3, 3)
    if data.ndim != 3 or data.shape[1:] != (3, 3):
        raise ValueError("data must have shape (n_frames, 3, 3)")

    # Validate data is not empty
    if data.shape[0] == 0:
        raise ValueError("Input data cannot be empty")

    return data.astype(np.float64, copy=False)


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


def _create_relative_matrices_and_postural_angles(
    data: npt.NDArray[np.float64]
) -> tuple[
    npt.NDArray[np.float64],
    PosturalAngles,
]:
    """
    Calculate relative rotations and postural angles.

    Args:
        data (npt.NDArray[np.float64]): Array of rotation matrices with shape (n_frames, 3, 3).

    Returns:
        tuple[npt.NDArray[np.float64], PosturalAngles]: A tuple containing:
            - Relative rotation matrices
            - Normalized postural angles
    """

    postural_angles = _get_postural_angles(data)
    postural_angles = _normalize_postural_angles(postural_angles)

    relative_matrices = _compute_incremental_rotation_matrices(
        data
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
    data: npt.NDArray[np.float64],
    postural_data: PosturalAngles,
    elevation_start: float,
    elevation_end: float,
    poe_start: float,
    poe_end: float,
) -> npt.NDArray[np.float64]:
    """Extract rows of data that fall within a specified elevation and POE bin.

    Args:
        data (npt.NDArray[np.float64]): An array of data that you wish to filter. Must be same length of postural_data.

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
        the specified elevation and POE bin. Will have the length of `len(data) - 1`,
        because the last frame is omitted for relative rotations.

    Raises:
        TypeError: If postural_data is not a PosturalAngles instance.
        ValueError: If postural_data has None values for poe or elevation.
        ValueError: If either bin has invalid bounds.
    """

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
    return data[elevation_mask & poe_mask]


def _calculate_trace_rotation_angles(
    rotation_matrices: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the absolute rotation angle for each relative rotation matrix using the trace formula.

    For each proper rotation matrix, the magnitude of the rotation is computed as:
        theta = arccos((trace(R) - 1) / 2)

    Args:
        rotation_matrices (npt.NDArray[np.float64]): Relative rotation matrices
            with shape (n_steps, 3, 3).

    Returns:
        npt.NDArray[np.float64]: Rotation magnitudes in radians, with shape (n_steps,).

        Raises:
            ValueError: If the input shape is invalid or empty.
    
    Notes:
        - The result is always positive and in the range [0, π].
        - Due to numerical precision, trace values outside [-1, 3] are
          clamped to a valid range before computing arccos.
    """
    matrices = np.asarray(rotation_matrices, dtype=np.float64)

    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError("rotation_matrices must have shape (n_steps, 3, 3)")

    if matrices.shape[0] == 0:
        raise ValueError("rotation_matrices must contain at least one matrix")

    traces = np.trace(matrices, axis1=1, axis2=2)
    cos_angles = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    return np.asarray(angles, dtype=np.float64)


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
    traces: npt.NDArray[np.float64],
    euler_components: npt.NDArray[np.float64],
    postural_angles: PosturalAngles,
    elevation_start: int,
    elevation_end: int,
    poe_start: int,
    poe_end: int,
) -> BinRotationResult:
    """Calculate cumulative motion metrics for one heatmap bin.

    Args:
        traces (npt.NDArray[np.float64]): Trace angles with shape (n_frames - 1,).
        euler_components (npt.NDArray[np.float64]): Euler components with shape (n_frames - 1, 3).
        postural_angles (PosturalAngles): Normalized postural angles for each frame.
        elevation_start (int): Lower elevation bound (inclusive).
        elevation_end (int): Upper elevation bound (exclusive).
        poe_start (int): Lower POE bound (inclusive).
        poe_end (int): Upper POE bound (exclusive).


    Returns:
        BinRotationResult: A data class containing the calculated metrics.
    """

    # Extract data from current bin
    bin_euler_data = _extract_bin_data(
        data=euler_components,
        postural_data=postural_angles,
        elevation_start=elevation_start,
        elevation_end=elevation_end,
        poe_start=poe_start,
        poe_end=poe_end,
    )
    bin_trace_data = _extract_bin_data(
        data=traces,
        postural_data=postural_angles,
        elevation_start=elevation_start,
        elevation_end=elevation_end,
        poe_start=poe_start,
        poe_end=poe_end,
    )

    # return zero values if no data in the bin
    if bin_euler_data.shape[0] == 0:
        return BinRotationResult(
            elevation=np.float64(0),
            poe=np.float64(0),
            ir_er=np.float64(0),
            cumulative_motion=np.float64(0),
            sample_count=0
        )

    # sum euler components to get cumulative motion for each axis
    elevation, poe, ir_er = (_accumulate_euler_components(bin_euler_data))

    # sum trace angles to get cumulative motion for the bin
    total_trace_motion = np.sum(bin_trace_data)

    # get the number of samples in the bin
    n_samples = bin_euler_data.shape[0]

    return BinRotationResult(
        elevation=elevation,
        poe=poe,
        ir_er=ir_er,
        cumulative_motion=np.float64(total_trace_motion),
        sample_count=n_samples,
    )


def _add_bin_result_to_heatmap(
    heatmap: Heatmap,
    result: BinRotationResult,
    elevation_index: int,
    poe_index: int,
) -> None:
    """Helper function to write a single bin result to a heatmap cell.

    Args:
        heatmap (Heatmap): The heatmap to update.
        result (BinRotationResult): The result to write.
        elevation_index (int): Row index for the elevation bin.
        poe_index (int): Column index for the POE bin.

    Returns:
        None: The heatmap is updated in place.
    """

    heatmap.elevation[elevation_index, poe_index] = result.elevation
    heatmap.poe[elevation_index, poe_index] = result.poe
    heatmap.ir_er[elevation_index, poe_index] = result.ir_er
    heatmap.cumulative_motion[elevation_index, poe_index] = result.cumulative_motion
    heatmap.sample_count[elevation_index, poe_index] = result.sample_count


def _populate_heatmap(
    relative_matrices: npt.NDArray[np.float64],
    postural_angles: PosturalAngles,
    heatmap: Heatmap,
    participant_idx: int,
) -> Heatmap:
    """Populate heatmap with rotational motion data.

    Args:
        relative_matrices (npt.NDArray[np.float64]): Relative rotation matrices with shape (n_frames - 1, 3, 3).
        postural_angles (PosturalAngles): Normalized postural angles for each frame.
        heatmap (Heatmap): Heatmap object to populate.

    Returns:
        Heatmap: Updated heatmap with calculated metrics for each bin.
    """

    n_elevation_bins, n_poe_bins = heatmap.shape

    # Pre-allocate 2D arrays so each bin result maps directly to one grid cell.
    heatmap.elevation = np.zeros((n_elevation_bins, n_poe_bins), dtype=np.float64)
    heatmap.poe = np.zeros((n_elevation_bins, n_poe_bins), dtype=np.float64)
    heatmap.ir_er = np.zeros((n_elevation_bins, n_poe_bins), dtype=np.float64)
    heatmap.cumulative_motion = np.zeros((n_elevation_bins, n_poe_bins), dtype=np.float64)
    heatmap.sample_count = np.zeros((n_elevation_bins, n_poe_bins), dtype=np.int32)

    # Calculate trace totals and Euler components for entire arm once instead of
    # running these calculations over again for each bin.
    all_trace_totals = _calculate_trace_rotation_angles(relative_matrices)
    all_euler_components = _decompose_rotation_matrices_yxy(relative_matrices)

    # Run the calcs for each bin
    for bin_bounds in _generate_heatmap_bins(
        heatmap.bin_width,
        heatmap.elevation_range_end,
        heatmap.poe_range_end
    ):
        result = _calculate_single_bin(
            traces=all_trace_totals,
            euler_components=all_euler_components,
            postural_angles=postural_angles,
            **bin_bounds.__dict__,
        )

        elevation_index = bin_bounds.elevation_start // heatmap.bin_width
        poe_index = bin_bounds.poe_start // heatmap.bin_width

        _add_bin_result_to_heatmap(
            heatmap,
            result,
            elevation_index,
            poe_index,
        )

        # update procress bar
        get_pbar_manager().update_inner(participant_idx)

    return heatmap


def calculate_bin_rotations(
    data: npt.NDArray[np.float64],
    arm: Literal['left', 'right'],
    participant_idx: int,
) -> Heatmap:   
    """
    Calculate the cumulative motion for each bin in the heatmap for a given arm.
    """
    # validate incoming data
    data_array = _validate_rotation_data(data)

    # calculate relative motion
    relative_matrices, postural_angles = _create_relative_matrices_and_postural_angles(data_array, arm)

    # Initialize heatmap object
    heatmap = Heatmap()

    # return populated heatmap
    return _populate_heatmap(relative_matrices, postural_angles, heatmap, participant_idx)
