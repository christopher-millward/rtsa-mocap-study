"""
Main module to run the full analysis pipeline for the study.
Author: Christopher Millward
"""
from pathlib import Path
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

        # clean and validate data
        clean_data = clean_and_validate_data(raw_data)

        # run kinematics
        right_arm_kinematics = calculate_bin_rotations(clean_data, 'right', i)
        left_arm_kinematics = calculate_bin_rotations(clean_data, 'left', i)

        # save heatmap data
        participant_details[i].right.humerothoracic.heatmap = right_arm_kinematics
        participant_details[i].left.humerothoracic.heatmap = left_arm_kinematics

        # save trace total rotation value
        participant_details[i].right.humerothoracic.trace_total = right_arm_kinematics.cumulative_motion.sum()
        participant_details[i].left.humerothoracic.trace_total = left_arm_kinematics.cumulative_motion.sum()

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
