# In Vivo RTSA Shoulder Motion Analysis

**Author:** Christopher Millward  
**Date:** May 2026

## Project Overview

This repository contains a reproducible Python pipeline for analyzing normalized motion-capture data from shoulder biomechanics studies.  
It computes participant-level humerothoracic rotation summaries, bin-wise rotation heatmaps, and stores the results in a serialized output file for downstream analysis.

## Reproduce the Results
It is highly recommended that you first create a conda environment for this repository. Once created and activated, run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

This will:

1. load participant metadata from `data/raw_normalized_data/participant_details.xlsx`,
2. load each participant’s motion-capture data,
3. clean and validate the data,
4. compute total humerothoracic rotation for the right and left arms,
5. compute bin-wise rotation heatmaps,
6. save the updated participant records to `outputs/all_results.pkl`.

To run the test suite:

```bash
pytest
```

## Project Structure

```text
.
├── config.py                 # Project paths and numerical tolerances
├── main.py                   # Entry point for the full analysis pipeline
├── modules/
│   ├── bin_calcs.py          # Bin-wise rotation calculations
│   ├── cumulative_rotation.py# Total rotation calculations
│   ├── data_loading.py       # Load participant metadata and motion-capture data
│   ├── data_preprocessing.py # Cleaning and validation
│   ├── data_saving.py        # Output persistence
│   └── general_utilities.py  # Shared helper functions
├── test/                     # Pytest-based validation suite
├── data/
│   ├── raw_data/
│   ├── raw_normalized_data/   # Input Excel file and normalized motion-capture data
│   └── temporary/
├── outputs/                  # Generated results
├── notebooks/
├── resources/
└── requirements.txt
```

## Analysis Pipeline

The pipeline is organized as follows:

1. **Load participant metadata**  
   `load_participant_details()` reads the Excel file containing participant identifiers and clinical metadata.

2. **Load motion-capture data**  
   `load_motion_capture_data()` loads the participant-specific kinematic data.

3. **Clean and validate**  
   `clean_and_validate_data()` prepares the data for analysis and enforces numerical consistency.

4. **Compute total rotation**  
   `calculate_total_rotation()` estimates the cumulative humerothoracic rotation for each arm.

5. **Compute bin-wise rotation**  
   `calculate_bin_rotations()` aggregates rotation into elevation/plane-of-elevation bins and produces the heatmap representation.

6. **Save results**  
   `save_data_to_pickle()` writes the final `ParticipantDetails` objects to `outputs/all_results.pkl`.

## Mathematical Foundation

The analysis operates on cleaned motion traces and summarizes them at two levels:

- **Global summary**: total humerothoracic rotation over the full movement trace.
- **Local summary**: cumulative rotation within fixed angular bins.

The binning scheme is defined by the `Heatmap` schema:

- elevation range: `0` to `180` degrees
- plane-of-elevation range: `0` to `360` degrees
- bin width: `20` degrees by default

This yields a grid in which each cell stores:

- cumulative elevation rotation,
- cumulative plane-of-elevation rotation,
- cumulative internal/external rotation,
- total cumulative motion,
- sample count.

Numerical validation uses strict tolerances defined in `config.py`, including:

- `ORTHONORMAL_TOLERANCE`
- `DETERMINANT_TOLERANCE`
- `SMALLEST_CLINICALLY_RELEVANT_ANGLE`
- `TEST_PRECISION_TOLERANCE`
- `TEST_SINGULARITY_TOLERANCE`

These thresholds help ensure that the rotation calculations are physically valid and numerically stable.

## Test-Driven Development

This project was developed using a test-driven workflow.

Each function was validated with tests in `test/` before development progressed, helping ensure:

- correct numerical behavior,
- robust handling of edge cases,
- stable file I/O,
- reproducible scientific outputs.

Test coverage includes:

- data loading,
- preprocessing,
- cumulative rotation,
- bin calculations,
- saving,
- general utility functions.

## Data Model

Results are stored using dataclasses defined in `schema.py`:

- `ParticipantDetails`
- `ArmRotationDetails`
- `RotationData`
- `Heatmap`

These structures make the results explicit, serializable, and easy to extend.

## Requirements

See `requirements.txt` for the full environment specification.

Main dependencies include:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `tqdm`
- `pytest`
- `openpyxl`

## Output

The main analysis writes:

- `outputs/all_results.pkl` — serialized participant results

A CSV export for heatmap data is prepared in the codebase but is not currently enabled in `main.py`.

## Notes

- The pipeline is intended to be run from the repository root.
- All major functions contain explanatory docstrings.
- The unit tests provide an audit trail for scientific correctness and implementation stability.