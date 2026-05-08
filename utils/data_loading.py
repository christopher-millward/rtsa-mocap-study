""" 
Functions for loading and processing participant data 

Author: Christopher Millward
"""

from typing import List, TypedDict, Literal, cast
import numpy as np
import numpy.typing as npt
import pandas as pd


class ArmRotationDetails(TypedDict):
    """Type definition for per-arm rotation summary metrics."""

    humerothoracic_rotation: float | None
    glenohumeral_rotation: float | None
    total_rotation_x: float | None
    total_rotation_y: float | None
    total_rotation_z: float | None


class ParticipantDetails(TypedDict):
    """Type definition for participant detail records.

    Attributes:
        filename (str): The participant's filename identifier.
        rtsa_side ('right' | 'left' | 'both' | None): Side of Reverse TSA procedure
            ('right', 'left', 'both', or None if no RTSA).
        tsa_side ('right' | 'left' | 'both' | None): Side of TSA procedure ('right', 'left', 'both', or 'none').
        dominant_arm ('right' | 'left' | None): Participant's dominant arm
            ('right', 'left', or None).
        age (int): Participant's age in years.
        left (ArmRotationDetails): Rotation summary metrics for the left arm.
        right (ArmRotationDetails): Rotation summary metrics for the right arm.
    """

    filename: str
    rtsa_side: Literal['right', 'left', 'both', None]
    tsa_side: Literal['right', 'left', 'both', None]
    dominant_arm: Literal['right', 'left', None]
    age: int
    left: ArmRotationDetails
    right: ArmRotationDetails


def _arms_for_side(side: Literal['right', 'left', 'both', None]) -> set[Literal['right', 'left']]:
    """Translate a side label into the affected arm labels."""
    if side == 'right':
        return {'right'}
    if side == 'left':
        return {'left'}
    if side == 'both':
        return {'right', 'left'}
    return set()


def load_participant_details(filepath: str) -> List[ParticipantDetails]:
    """Load participant details from an Excel file and return structured data.

    This function reads a participant details Excel file and returns a list of
    dictionaries containing key variables for each participant, including their
    filename, RTSA side (right/left shoulder arthroplasty), TSA side (total
    shoulder arthroplasty), dominant arm, and age.

    Args:
        filepath (str): Path to the participant_details.xlsx file.

    Returns:
        List[ParticipantDetails]: A list of dictionaries, where each dictionary
            represents a participant row with ParticipantDetails keys.

    Raises:
        FileNotFoundError: If the Excel file does not exist.
        pd.errors.EmptyDataError: If the Excel file is empty.
        KeyError: If expected columns are missing from the Excel file.

    Example:
        >>> participants = load_participant_details('./raw_data/participant_details.xlsx')
        >>> print(participants[0])
        {'filename': '1_R_MATRICES 2016-5-10', 'rtsa_side': 'right',
         'tsa_side': None, 'dominant_arm': 'right', 'age': 74}
    """
    df: pd.DataFrame = pd.read_excel(filepath)

    participants: List[ParticipantDetails] = []

    for _, row in df.iterrows():
        # Determine RTSA side
        if row.get('RTSA-R') == 1 and row.get('RTSA-L') == 1:
            rtsa_side = 'both'
        elif row.get('RTSA-R') == 1:
            rtsa_side = 'right'
        elif row.get('RTSA-L') == 1:
            rtsa_side = 'left'
        else:
            rtsa_side = None

        # Determine TSA side
        if row.get('TSA-R') == 1 and row.get('TSA-L') == 1:
            tsa_side = 'both'
        elif row.get('TSA-R') == 1:
            tsa_side = 'right'
        elif row.get('TSA-L') == 1:
            tsa_side = 'left'
        else:
            tsa_side = None

        # Determine dominant arm
        r_dom = row.get('R-DOM') == 1
        l_dom = row.get('L-DOM') == 1

        if r_dom == l_dom:
            raise ValueError(
                'Each participant must have exactly one dominant arm flag set.'
            )
        if r_dom:
            dominant_arm = 'right'
        else:
            dominant_arm = 'left'

        # Make sure RTSA and TSA are not on the same arm
        if _arms_for_side(rtsa_side) & _arms_for_side(tsa_side):
            """checks for an intersection between the two sets"""
            raise ValueError(
                'A participant cannot have RTSA and TSA on the same arm.'
            )

        participant: ParticipantDetails = {
            'filename': cast(str, row.get('fname')),
            'rtsa_side': rtsa_side,
            'tsa_side': tsa_side,
            'dominant_arm': dominant_arm,
            'age': cast(int, row.get('Age')),
            'left': {
                'humerothoracic_rotation': None,
                'glenohumeral_rotation': None,
                'total_rotation_x': None,
                'total_rotation_y': None,
                'total_rotation_z': None,
            },
            'right': {
                'humerothoracic_rotation': None,
                'glenohumeral_rotation': None,
                'total_rotation_x': None,
                'total_rotation_y': None,
                'total_rotation_z': None,
            },
        }

        participants.append(participant)

    return participants


def load_motion_capture_data(
    filename: str,
    data_dir: str = './raw_data',
) -> npt.NDArray[np.float64]:
    """Load motion-capture data from a tab-delimited file into a NumPy array.

    Args:
        filename (str): Motion-capture filename in the expected format from `ParticipantDetails['filename']`.
        data_dir (str): Directory that contains the raw motion-capture files.

    Returns:
        npt.NDArray[np.float64]: Array of motion-capture values loaded with `np.loadtxt`.
    """
    filepath = f'{data_dir}/{filename}'
    return np.loadtxt(
        filepath,
        delimiter='\t',
        skiprows=1,
        usecols=range(1, 19),
    )
