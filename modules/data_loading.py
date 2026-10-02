""" 
Functions for loading data.

Author: Christopher Millward
"""

from typing import List, Literal, cast
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from datetime import datetime, time
from config import RAW_DATA_DIR
from schema import ParticipantDetails, ArmRotationDetails, RotationData


def _arms_for_side(side: Literal['right', 'left', 'both', None]) -> set[Literal['left', 'right']]:
    """Translate a side label into the affected arm labels."""
    if side == 'right':
        return {'right'}
    if side == 'left':
        return {'left'}
    if side == 'both':
        return {'right', 'left'}
    return set()


def load_participant_details(filepath: str | Path) -> List[ParticipantDetails]:
    """Load participant details from an Excel file and return structured data.

    This function reads a participant details Excel file and returns a list of
    dictionaries containing key variables for each participant, including their
    filename, RTSA side (right/left shoulder arthroplasty), TSA side (total
    shoulder arthroplasty), dominant arm, and age.

    Args:
        filepath (str | Path): Path to the participant_details.xlsx file.

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
    filepath = Path(filepath)
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

        fname = cast(Path, row.get('fname'))
        participant: ParticipantDetails = ParticipantDetails(
            filename=fname,
            rtsa_side=rtsa_side,
            tsa_side=tsa_side,
            dominant_arm=dominant_arm,
            rtsa_on_dominant_arm=(
                None
                if rtsa_side is None
                else dominant_arm in _arms_for_side(rtsa_side)
            ),
            age=cast(int, row.get('Age')),
            sex=cast(str, row.get('Sex')),
            start_time=_get_first_entry_time(RAW_DATA_DIR / fname),
            left=ArmRotationDetails(
                humerothoracic=RotationData(),
                glenohumeral=RotationData(),
            ),
            right=ArmRotationDetails(
                humerothoracic=RotationData(),
                glenohumeral=RotationData(),
            ),
        )

        participants.append(participant)

    return participants


def _get_first_entry_time(file_path: str | Path) -> time:
    """Extract the time from the first data entry in a tab-delimited file.

    The file is expected to contain a header row and datetime values in the
    first column using the format ``YYYY-MM-DD H:MM:SS AM/PM``.

    Args:
        file_path: Path to the tab-delimited data file.

    Returns:
        The time component of the datetime in the first data row.

    Raises:
        ValueError: If the datetime value cannot be parsed.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        next(file)  # Skip header row

        first_row = next(file).strip().split("\t")
        datetime_value = datetime.strptime(
            first_row[0],
            "%Y-%m-%d %I:%M:%S %p",
        )

    return datetime_value.time()


def load_motion_capture_data(
    filename: str | Path,
    data_dir: str | Path = RAW_DATA_DIR,
) -> npt.NDArray[np.float64]:
    """Load motion-capture data from a tab-delimited file into a NumPy array.

    Args:
        filename (str | Path): Motion-capture filename in the expected format from `ParticipantDetails['filename']`.
        data_dir (str | Path): Directory that contains the raw motion-capture files.

    Returns:
        npt.NDArray[np.float64]: Array of motion-capture values loaded with `np.loadtxt`.
    """
    data_dir = Path(data_dir)
    filepath = data_dir / Path(filename)
    return np.loadtxt(
        filepath,
        delimiter='\t',
        skiprows=1,
        usecols=range(1, 19),
    )
