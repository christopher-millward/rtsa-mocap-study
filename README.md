# In Vivo RTSA Shoulder Motion Analysis

**Author:** Christopher Millward  
**Date:** May 2026

## Project Overview

This project analyzes three-dimensional shoulder kinematics data from a motion capture study. The goal is to quantify total rotational motion in the shoulders of participants with Reverse Total Shoulder Arthroplasty (RTSA) implants.

The analysis reads raw rotation matrices (3x3) and computes accumulated rotation angles, in radians, using the [trace formula](https://en.wikipedia.org/wiki/Trace_\(linear_algebra\)).

---

## Mathematical Foundation

### Rotation Matrices and Angle Calculation

Each frame of motion capture data contains two 3×3 rotation matrices (one for each arm). These matrices encode the orientation of the humerus relative to a sensor placed on the sternum.

The rotation angle **θ** for a given rotation matrix **R** is calculated from the trace (sum of diagonal elements):

$$\theta = \arccos\left(\frac{\text{trace}(R) - 1}{2}\right)$$

**Key properties:**
- The result is always in the range [0, π] radians
- Trace values are clamped to [−1, 3] before arccos to handle numerical precision issues
- Total rotation is the sum of all rotation angles across all frames

### Input Data Format

Each motion capture file contains one row per frame with 18 numeric columns:
- Column 0: Datetime stamp
- Columns 1–9: Left arm rotation matrix elements (row-major: L00, L01, ..., L22)
- Columns 10–18: Right arm rotation matrix elements (row-major: R00, R01, ..., R22)

---

## Repository Structure

```
New Analysis/
├── README.md                          # This file
├── pytest.ini                         # pytest configuration
├── dev.ipynb                          # Jupyter notebook for development and testing
├── main.py                            # (placeholder) Main analysis script
│
├── utils/                             # Utility modules
│   ├── data_loading.py                # Load participant metadata (Excel) and motion data (TSV)
│   │
│   ├── kinematics/                    # Shoulder kinematics calculations
│   │    ├── cumulative.py             # Calculate cumulative rotation across all axes (single value)
│   │    └── individual_axes.py        # Calculate rotation about each axis individually
│   │
│   └── visualizations.py              # (placeholder) Plotting functions
│
├── test/                              # Unit tests
│   ├── test_data_loading.py           # Tests for participant loading and motion data I/O
│   │
│   └── kinematics/                    # Shoulder kinematics calculations
│        ├── test_cumulative.py        # Tests for cumulative motion functions
│        └── test_individual_axes.py   # Tests for individual axis rotation functions
│
├── raw_data/
│   ├── participant_details.xlsx       # Metadata: participant info, RTSA/TSA status, age, dominance
│   └── 1_R_MATRICES .../              # Motion capture files (one per test session)
│
└── outputs/                           # Results and generated artifacts
```

---

## Key Modules and Functions

### `utils/data_loading.py`

**`load_participant_details(filepath)`**
- Reads an Excel file containing participant metadata.
- Returns a list of `ParticipantDetails` dictionaries.
- Validates that each participant has exactly one dominant-arm flag and no RTSA/TSA overlap on the same arm.

**`load_motion_capture_data(filename, data_dir='./raw_data')`**
- Loads motion capture data from a tab-delimited file (as a NumPy array).
- Expects 18 columns (left and right arm rotation matrices).
- Skips the first row (header).

### `utils/kinematics.py`

**`extract_rotation_matrix(row, arm)`**
- Extracts a 3×3 rotation matrix from a 1D row of motion data.
- `arm` is either `'L'` or `'R'`.

**`calculate_rotation_angle(rotation_matrix)`**
- Computes the rotation angle in radians for a single 3×3 matrix using the trace formula.

**`calculate_total_rotation(data, arm)`**
- Vectorized computation of total rotation for an arm across all frames.
- `data` is a 2D NumPy array (n_frames × 18).
- Returns the sum of all rotation angles in radians.

**`calculate_arm_rotations(data)`**
- High-level function that computes total rotation for both arms.
- Returns `(left_rotation, right_rotation)` in radians.

---

## Getting Started

### Setup

1. **Install dependencies:**
   ```bash
   pip install pandas numpy openpyxl pytest
   ```

2. **Run tests to verify installation:**
   ```bash
   pytest
   ```

### Usage Example

In the `dev.ipynb` notebook:

```python
from utils.data_loading import load_participant_details, load_motion_capture_data
from utils.kinematics import calculate_arm_rotations

# Load participant metadata
participants = load_participant_details('./raw_data/participant_details.xlsx')

# Get the first participant's filename
participant = participants[0]
filename = participant['filename']

# Load motion capture data for that participant
data = load_motion_capture_data(filename)

# Calculate total rotation for both arms
left_rotation, right_rotation = calculate_arm_rotations(data)

print(f"Left: {left_rotation:.2f} rad, Right: {right_rotation:.2f} rad")
```

---

## Testing

All code is tested using **pytest** with verbose output. To run tests:

```bash
# Run all tests
pytest

# Run a specific test file
pytest test/test_kinematics.py

# Run with verbose output (default)
pytest -v
```

The test suite includes:
- **Unit tests for data loading:** Participant metadata extraction, file I/O, validation rules.
- **Unit tests for kinematics:** Rotation matrix extraction, angle calculation, accumulation across frames.

Tests use mocked I/O to keep them fast and deterministic (no dependency on actual files).

---

## Input Data Formats

### Participant Details (Excel)

Columns:
- `fname`: Participant filename identifier
- `RTSA-R`, `RTSA-L`: Binary flags (1 = yes, 0 = no) for Reverse TSA on right/left
- `TSA-R`, `TSA-L`: Binary flags for total TSA
- `R-DOM`, `L-DOM`: Binary flags for dominant arm (exactly one must be 1)
- `Age`: Participant age in years

### Motion Capture Data (Tab-Delimited)

- **First row:** Header (skipped during load)
- **Remaining rows:** One frame per row
- **Columns 1–18:** Datetime stamp (skipped) + 18 numeric values for rotation matrices

---

## Next Steps / TODO
- Calculate amount of motion about each axis
   - Wrote 2 versions of the decomposition function. One using Scipy.spatial, which saves a lot of code. 
      - Review errors in the testing suite to see where logic breaks down / what the boundaries are. 
- Add explanation of individual axis claclulations to README mathematical foundation section
- Segment location binning (identify which region each frame is within (elev + POE))
   - Use this to determine total magnitude 
- Build scapular correction module (can switch out module logic later)
- Visualization functions (trajectories, heatmaps, statistical plots)
- Statistical analysis and export