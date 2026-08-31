"""
Main module to run the full analysis pipeline for the study.
Author: Christopher Millward
"""
from pathlib import Path
from typing_extensions import Literal, cast
from modules.general_utilities import create_rotation_matrices
from modules.kinematics import calculate_bin_rotations
from modules.data_loading import load_participant_details, load_motion_capture_data
from modules.data_preprocessing import clean_and_validate_data
from modules.data_saving import save_data_to_pickle
from config import RAW_PARTICIPANT_DETAILS_PATH, RESULTS_PICKLE_PATH
from modules.progress_bar import initialize_pbar, get_pbar_manager
from modules.statistics import run_statistics
from modules.plots import create_and_save_all_figures


def main():
    """This function orchestrates the entire analysis pipeline."""

    # Load participant details from the Excel file
    participant_details = load_participant_details(RAW_PARTICIPANT_DETAILS_PATH)

    # remove all files with no RTSA
    participant_details = [participant for participant in participant_details if participant.rtsa_side is not None]

    # Set up progress bar
    initialize_pbar(participant_details) 

    # For each participant
    for i, participant in enumerate(participant_details):

        # load the data
        raw_data = load_motion_capture_data(participant.filename)

        for side in ['left', 'right']:
            # appease the type checker
            side = cast(Literal["left", "right"], side)

            # create R matrices
            data = create_rotation_matrices(raw_data, side)

            # clean and validate data
            cleaned_data = clean_and_validate_data(data)

            # run kinematics
            kinematics = calculate_bin_rotations(cleaned_data, side, i)

            # save kinematics data
            arm = getattr(participant_details[i], side)
            arm.humerothoracic.heatmap = kinematics
            arm.humerothoracic.trace_total = kinematics.cumulative_motion.sum()

        # update progress bar
        get_pbar_manager().update_outer()

    # Save data 
    save_data_to_pickle(participant_details, Path(RESULTS_PICKLE_PATH))

    # Run and save statistics
    run_statistics(participant_details)
    create_and_save_all_figures(participant_details)

    # Close progress bar
    get_pbar_manager().close()

if __name__ == "__main__":
    main()
