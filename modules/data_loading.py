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

@dataclass(slots=True)
class Heatmap:
    """
    Type definition for a heatmap.

    Each of elevation, poe, and ir_er is a 2D array representing the cumulative 
    motion in each bin of the heatmap. The shape of these arrays is determined 
    by the bin_width, which must evenly divide 360 degrees.

    Attributes:
        bin_width: Width of each heatmap bin in degrees.
        elevation: Cumulative elevation motion.
        poe: Cumulative plane-of-elevation motion.
        ir_er: Cumulative internal/external rotation motion.
        cumulative_motion: Total cumulative motion.
        sample_count: Number of samples contributing to each bin.
    """

    bin_width: float = 20.0

    elevation: npt.NDArray[np.float64] | None = None
    poe: npt.NDArray[np.float64] | None = None
    ir_er: npt.NDArray[np.float64] | None = None

    cumulative_motion: npt.NDArray[np.float64] | None = None

    sample_count: npt.NDArray[np.int32] | None = None

    def __post_init__(self) -> None:
        if 360 % self.bin_width != 0:
            raise ValueError(
                f"bin_width ({self.bin_width}) must evenly divide 360."
            )

        n_bins = int(360 / self.bin_width)
        shape = (n_bins, n_bins)

        if self.elevation is None:
            self.elevation = np.zeros(shape, dtype=np.float64)

        if self.poe is None:
            self.poe = np.zeros(shape, dtype=np.float64)

        if self.ir_er is None:
            self.ir_er = np.zeros(shape, dtype=np.float64)

        if self.cumulative_motion is None:
            self.cumulative_motion = np.zeros(shape, dtype=np.float64)

        if self.sample_count is None:
            self.sample_count = np.zeros(shape, dtype=np.int32)

@dataclass
class RotationData:
    trace_total: np.float64 | None = None
    heatmap: Heatmap = field(default_factory=Heatmap)

@dataclass
class ArmRotationDetails():
    """Type definition for per-arm rotation summary metrics.

    Attributes:
        trace_total (np.float64 | None): The total rotation for the arm.
        rotation_bins (RotationBins | None): The rotation bins for the arm.
    """

    humerothoracic: RotationData = field(default_factory=RotationData)
    glenohumeral: RotationData = field(default_factory=RotationData)


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
