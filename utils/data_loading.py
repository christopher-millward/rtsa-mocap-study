""" 
Functions for loading and processing participant data 

Author: Christopher Millward
"""

from typing import List, TypedDict, Literal, cast
import pandas as pd


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
        left_rotation (float | None): Total accumulated rotation for left arm in radians.
        right_rotation (float | None): Total accumulated rotation for right arm in radians.
    """

    filename: str
    rtsa_side: Literal['right', 'left', 'both', None]
    tsa_side: Literal['right', 'left', 'both', None]
    dominant_arm: Literal['right', 'left', None]
    age: int
    left_rotation: float | None
    right_rotation: float | None


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
        if row.get('R-DOM') == 1:
            dominant_arm = 'right'
        elif row.get('L-DOM') == 1:
            dominant_arm = 'left'
        else:
            dominant_arm = None

        participant: ParticipantDetails = {
            'filename': cast(str, row.get('fname')),
            'rtsa_side': rtsa_side,
            'tsa_side': tsa_side,
            'dominant_arm': dominant_arm,
            'age': cast(int, row.get('Age')),
            'left_rotation': None,
            'right_rotation': None,
        }

        participants.append(participant)

    return participants
