"""Functions for helper functions used in the kinematics pipeline.

Author: Christopher Millward
"""
import numpy as np
import numpy.typing as npt
from schema import Heatmap, ParticipantDetails


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


def convert_heatmap_to_degrees(heatmap: Heatmap) -> Heatmap:
    """Convert kinematics values in a heatmap from radians to degrees.

    Args:
        heatmap (Heatmap): The input heatmap with values in radians.

    Returns:
        Heatmap: The output heatmap with values in degrees.
    """

    deg_heatmap = Heatmap(
        bin_width=heatmap.bin_width,
        elevation_range_end=heatmap.elevation_range_end,
        poe_range_end=heatmap.poe_range_end,
        elevation=np.degrees(heatmap.elevation),
        poe=np.degrees(heatmap.poe),
        ir_er=np.degrees(heatmap.ir_er),
        cumulative_motion=np.degrees(heatmap.cumulative_motion),
        sample_count=heatmap.sample_count
    )

    return deg_heatmap


def convert_all_heatmaps_to_degrees(
    participant_details: list[ParticipantDetails],
) -> list[ParticipantDetails]:
    """Convert all kinematics heatmaps from radians to degrees in place.

    NOTE:   At this time, the function only modified the humerothoracic heatmap 
            for each arm. If additional heatmaps are added to the 
            ParticipantDetails schema, this function will need to be updated 
            accordingly.

    Args:
        participant_details: Participant data containing heatmaps in radians.

    Returns:
        The same ParticipantDetails objects, with heatmaps converted to degrees.
    """
    for participant in participant_details:
        for side in ["left", "right"]:
            arm = getattr(participant, side)
            arm.humerothoracic.heatmap = convert_heatmap_to_degrees(
                arm.humerothoracic.heatmap
            )

    return participant_details
