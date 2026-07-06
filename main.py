"""
Main module to run the full analysis pipeline for the study.
Author: Christopher Millward
"""
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Any, cast
from modules.data_loading import load_participant_details, load_motion_capture_data
from modules.data_preprocessing import clean_and_validate_data
from modules.general_utilities import create_rotation_matrices
from modules.cumulative_rotation import calculate_total_rotation
from config import RAW_DATA_DIR, RESULTS_PATH

# ---- Vars ----
details_path = RAW_DATA_DIR / "participant_details.xlsx"


def main():
    """This function orchestrates the entire analysis pipeline."""

    # Load participant details from the Excel file
    participant_details = load_participant_details(details_path)

    # Set up progress bar
    progress = tqdm(
        enumerate(participant_details),
        total=len(participant_details),
        desc="Processing participants",
        unit="participant",
    )

    # For each participant
    for i, participant in progress:
        progress.set_postfix(file=participant["filename"])

        # load the data
        raw_data = load_motion_capture_data(participant['filename'])

        # skip file if there is no RTSA
        if participant['rtsa_side'] is None:
            continue

        # Build R matrices
        data_right = create_rotation_matrices(raw_data, arm='right')
        data_left = create_rotation_matrices(raw_data, arm='left')

        # clean data
        data_right = clean_and_validate_data(data_right, 'right')
        data_left = clean_and_validate_data(data_left, 'left')

        # run analysis
        # cumulative rotation
        total_right = calculate_total_rotation(data_right, 'right')
        total_left = calculate_total_rotation(data_left, 'left')

        # rotation in each bin
        # i.e., how much flex/ext in the range of 0-10 degrees, 10-20 degrees, etc.
        # do this for each axis (flex/ext, abd/add, int/ext rotation)

        # Save the results to the data object
        participant_details[i]['right']['humerothoracic_rotation'] = total_right
        participant_details[i]['left']['humerothoracic_rotation'] = total_left
        # save bin calcs

    # write data object to a CSV file
    df = pd.json_normalize(
        cast(list[dict[str, Any]], participant_details),
        sep="_",
    )
    df.to_csv(RESULTS_PATH, index=False)

if __name__ == "__main__":
    main()
