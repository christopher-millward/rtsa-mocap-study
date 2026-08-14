"""
Main module to run the full analysis pipeline for the study.
Author: Christopher Millward
"""
from tqdm import tqdm
from pathlib import Path
from modules.bin_calcs import calculate_bin_rotations
from modules.data_loading import load_participant_details, load_motion_capture_data
from modules.data_preprocessing import clean_and_validate_data
from modules.cumulative_rotation import calculate_total_rotation
from modules.data_saving import save_data_to_pickle
from config import RAW_PARTICIPANT_DETAILS_PATH, RESULTS_PICKLE_PATH


def main():
    """This function orchestrates the entire analysis pipeline."""

    # Load participant details from the Excel file
    participant_details = load_participant_details(RAW_PARTICIPANT_DETAILS_PATH)

    # remove all files with no RTSA
    participant_details = [participant for participant in participant_details if participant.rtsa_side is not None]

    # Set up progress bar
    progress = tqdm(
        enumerate(participant_details),
        total=len(participant_details),
        desc="Processing participants",
        unit="participant",
    )

    # For each participant
    for i, participant in progress:
        progress.set_postfix(file=participant.filename)

        # load the data
        raw_data = load_motion_capture_data(participant.filename)

        # clean and validate data
        clean_data = clean_and_validate_data(raw_data)

        # cumulative rotation
        # calculate
        total_right = calculate_total_rotation(clean_data, 'right')
        total_left = calculate_total_rotation(clean_data, 'left')
        # save
        participant_details[i].right.humerothoracic.trace_total = total_right
        participant_details[i].left.humerothoracic.trace_total = total_left


        # rotation in each bin
        # calculate
        right_bin_calcs = calculate_bin_rotations(clean_data, 'right')
        left_bin_calcs = calculate_bin_rotations(clean_data, 'left')
        # save
        participant_details[i].right.humerothoracic.heatmap = right_bin_calcs
        participant_details[i].left.humerothoracic.heatmap = left_bin_calcs


    # Save data 
    save_data_to_pickle(participant_details, Path(RESULTS_PICKLE_PATH))
    # save_heatmap_to_csv(participant_details, Path(RESULTS_HEATMAP_CSV_PATH))

if __name__ == "__main__":
    main()
