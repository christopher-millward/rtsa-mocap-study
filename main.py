"""
Main module to run the full analysis pipeline for the study.
Author: Christopher Millward
"""
from pathlib import Path
from modules.data_loading import load_participant_details, load_motion_capture_data
from modules.data_cleaning import clean_data

### Vars
details_path = Path.cwd() / "data" / "raw_normalized_data" / "participant_details.xlsx"



def main():
# Main function to run the analysis
    
    # Load participant details from the Excel file
    participant_details = load_participant_details(details_path)

    # For each participant
    for participant in participant_details:
        # load the data
        raw_data = load_motion_capture_data(participant['filename'])

        # clean data
        # cleaned_data = clean_data(raw_data)

        # run analysis
            # for each arm (op and non-op)
                # cumulative motion (all axes)
                # motion about each axis (in degrees)
                # how much rotation in each bin
                    # i.e., how much flex/ext in the range of 0-10 degrees, 10-20 degrees, etc.
                    # do this for each axis (flex/ext, abd/add, int/ext rotation)
        # Save the results to a data object
    # write data object to a CSV file
    return NotImplementedError

if __name__ == "__main__":
    main()