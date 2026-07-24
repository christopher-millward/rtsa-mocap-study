""" 
Functions for loading data.

Author: Christopher Millward
"""

from dataclasses import dataclass, field
from typing import List, TypedDict, Literal, cast, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
from pathlib import Path
from config import RAW_DATA_DIR

@dataclass()
class SingleBin:
    """Type definition for a single rotation bin.

    Attributes:
        min (int): The minimum value of the rotation bin.
        max (int): The maximum value of the rotation bin.
        elevation (np.float64 | None): The elevation of the rotation bin.
        ir_er (np.float64 | None): The IR/ER value of the rotation bin.
    """
    min: int 
    max: int
    elevation: np.float64 | None = None
    ir_er: np.float64 | None = None

@dataclass
class RotationBins:
    """
    Type definition for rotation bins.

    A single bin has the attributes:
        - min (int): The minimum value of the rotation bin.
        - max (int): The maximum value of the rotation bin.
        - elevation (np.float64 | None): The elevation of the rotation bin.
        - ir_er (np.float64 | None): The IR/ER value of the rotation bin.

    Attributes:
        rng_0_20 (SingleBin): Rotation range 0-20 degrees.
        rng_20_40 (SingleBin): Rotation range 20-40 degrees.
        rng_40_60 (SingleBin): Rotation range 40-60 degrees.
        rng_60_80 (SingleBin): Rotation range 60-80 degrees.
        rng_80_100 (SingleBin): Rotation range 80-100 degrees.
        rng_100_120 (SingleBin): Rotation range 100-120 degrees.
        rng_120_140 (SingleBin): Rotation range 120-140 degrees.
        rng_140_160 (SingleBin): Rotation range 140-160 degrees.
        rng_160_180 (SingleBin): Rotation range 160-180 degrees.
    """
    range: tuple[int, int] = (0, 180)
    step: int = 20
    rng_0_20: SingleBin = field(default_factory=lambda: SingleBin(0, 20))
    rng_20_40: SingleBin = field(default_factory=lambda: SingleBin(20, 40))
    rng_40_60: SingleBin = field(default_factory=lambda: SingleBin(40, 60))
    rng_60_80: SingleBin = field(default_factory=lambda: SingleBin(60, 80))
    rng_80_100: SingleBin = field(default_factory=lambda: SingleBin(80, 100))
    rng_100_120: SingleBin = field(default_factory=lambda: SingleBin(100, 120))
    rng_120_140: SingleBin = field(default_factory=lambda: SingleBin(120, 140))
    rng_140_160: SingleBin = field(default_factory=lambda: SingleBin(140, 160))
    rng_160_180: SingleBin = field(default_factory=lambda: SingleBin(160, 180))

@dataclass
class ArmRotationDetails():
    """Type definition for per-arm rotation summary metrics.

    Attributes:
        total_humerothoracic_rotation (np.float64 | None): Description of metric.
        total_glenohumeral_rotation (np.float64 | None): Description of metric.
        rotation_bins (RotationBins | None): The rotation bins for the arm.
    """

    total_humerothoracic_rotation: np.float64 | None
    total_glenohumeral_rotation: np.float64 | None
    rotation_bins: RotationBins

@dataclass
class ParticipantDetails():
    """Type definition for participant detail records.

    Attributes:
        filename (Path): The participant's filename identifier.
        rtsa_side ('right' | 'left' | 'both' | None): Side of Reverse TSA procedure
            ('right', 'left', 'both', or None if no RTSA).
        tsa_side ('right' | 'left' | 'both' | None): Side of TSA procedure ('right', 'left', 'both', or 'none').
        dominant_arm ('right' | 'left' | None): Participant's dominant arm
            ('right', 'left', or None).
        age (int): Participant's age in years.
        left (ArmRotationDetails): Rotation summary metrics for the left arm.
        right (ArmRotationDetails): Rotation summary metrics for the right arm.
    """

    filename: Path
    rtsa_side: Literal['right', 'left', 'both', None]
    tsa_side: Literal['right', 'left', 'both', None]
    dominant_arm: Literal['right', 'left', None]
    age: int
    left: ArmRotationDetails
    right: ArmRotationDetails


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

        participant: ParticipantDetails = ParticipantDetails(
            filename=cast(Path, row.get('fname')),
            rtsa_side=rtsa_side,
            tsa_side=tsa_side,
            dominant_arm=dominant_arm,
            age=cast(int, row.get('Age')),
            left=ArmRotationDetails(
                total_humerothoracic_rotation=None,
                total_glenohumeral_rotation=None,
                rotation_bins=RotationBins(),
            ),
            right=ArmRotationDetails(
                total_humerothoracic_rotation=None,
                total_glenohumeral_rotation=None,
                rotation_bins=RotationBins(),
            ),
        )

        participants.append(participant)

    return participants


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
