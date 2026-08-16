"""Schema for data structures used in this repo."""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
import numpy.typing as npt
from pathlib import Path

# ----------------------------
# Participant data model
# ----------------------------
@dataclass(slots=True)
class Heatmap:
    """
    Type definition for a heatmap.

    Each of elevation, poe, and ir_er is a 2D array (shape n_elev_bins, n_poe_bins) 
    representing the cumulative motion in each bin (grid cell) of the heatmap. 
    The shape of these arrays is determined by the bin_width, which must evenly 
    divide 360 degrees.

    Attributes:
        bin_width (int): Width of each heatmap bin in degrees.
        elevation_range_end (int): End of the elevation range in degrees.
        poe_range_end (int): End of the plane-of-elevation range in degrees.
        elevation (npt.NDArray[np.float64]): Array of cumulative elevation motion.
        poe (npt.NDArray[np.float64]): Array of cumulative plane-of-elevation motion.
        ir_er (npt.NDArray[np.float64]): Array of cumulative internal/external rotation motion.
        cumulative_motion (npt.NDArray[np.float64]): Array of total cumulative motion.
        sample_count (npt.NDArray[np.int32]): Number of samples contributing to each bin.
    """

    bin_width: int = 20
    elevation_range_end: int = 180
    poe_range_end: int = 360

    elevation: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    poe: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    ir_er: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    cumulative_motion: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    sample_count: npt.NDArray[np.int32] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.int32)
    )

    def __post_init__(self) -> None:
        if self.elevation_range_end % self.bin_width != 0:
            raise ValueError(
                f"bin_width ({self.bin_width}) must evenly divide {self.elevation_range_end}."
            )

        if self.poe_range_end % self.bin_width != 0:
            raise ValueError(
                f"bin_width ({self.bin_width}) must evenly divide {self.poe_range_end}."
            )
        
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Heatmap):
            return NotImplemented

        return (
            self.bin_width == other.bin_width
            and self.elevation_range_end == other.elevation_range_end
            and self.poe_range_end == other.poe_range_end
            and np.array_equal(self.elevation, other.elevation)
            and np.array_equal(self.poe, other.poe)
            and np.array_equal(self.ir_er, other.ir_er)
            and np.array_equal(self.cumulative_motion, other.cumulative_motion)
            and np.array_equal(self.sample_count, other.sample_count)
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the heatmap arrays."""
        n_elev_bins = self.elevation_range_end // self.bin_width
        n_poe_bins = self.poe_range_end // self.bin_width
        return (n_elev_bins, n_poe_bins)

@dataclass
class RotationData:
    """Type definition for rotation data.

    Attributes:
        trace_total (np.float64 | None): The total rotation for the arm.
        heatmap (Heatmap): The heatmap for the arm.
    """
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
        rtsa_side ('right' | 'left' | 'both' | None): Side of Reverse TSA procedure ('right', 'left', 'both', or None if no RTSA).
        tsa_side ('right' | 'left' | 'both' | None): Side of TSA procedure ('right', 'left', 'both', or 'none').
        dominant_arm ('right' | 'left' | None): Participant's dominant arm ('right', 'left', or None).
        age (int): Participant's age in years.
        left (ArmRotationDetails): Rotation summary metrics for the left arm.
        right (ArmRotationDetails): Rotation summary metrics for the right arm.
        operated (list[ArmRotationDetails]): A list of the rotation summary metrics for the operated arm(s).
        non_operated (list[ArmRotationDetails]): A list of the rotation summary metrics for the non-operated arm(s).
    """

    filename: Path
    rtsa_side: Literal['right', 'left', 'both', None]
    tsa_side: Literal['right', 'left', 'both', None]
    dominant_arm: Literal['right', 'left', None]
    age: int
    left: ArmRotationDetails
    right: ArmRotationDetails

    @property
    def operated(self) -> list[ArmRotationDetails]:
        """Return the operated arm(s)."""
        match self.rtsa_side:
            case "left":
                return [self.left]
            case "right":
                return [self.right]
            case "both":
                return [self.left, self.right]
            case None:
                return []

    @property
    def non_operated(self) -> list[ArmRotationDetails]:
        """Return the non-operated arm(s)."""
        match self.rtsa_side:
            case "left":
                return [self.right]
            case "right":
                return [self.left]
            case "both" | None:
                return []