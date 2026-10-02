"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

Plan:
I am going to run both paired t-tests and mixed effects models to compare both:
- rotation rate
- difference in rotation rate between op and non-op arms

while controlling for:
- age
- sex
- dominant arm

In order to do this, the data will need to be structured in a long format, with
each row representing a single arm for a single participant. Each row should 
have the following columns:
- participant: The participant's identifier.
- arm: The arm (operated or non-operated).
- rotation rate: The rotation rate for the arm.
- age: The participant's age.
- sex: The participant's sex.
- dominant arm: Whether or not this is the participant's dominant arm (boolean).
"""
from pathlib import Path

import pandas as pd
from scipy import stats
from config import STATISTICS_OUTPUT_PATH, RESULTS_PICKLE_PATH
from schema import ParticipantDetails

# -------------------------------------------------------------------
# Constants for column names in the statistics DataFrame
# -------------------------------------------------------------------
OPERATED_CUMULATIVE_ROTATION = "Operated cumulative rotation"
NON_OPERATED_CUMULATIVE_ROTATION = "Non-operated cumulative rotation"
OPERATED_ROTATION_RATE = "Operated rotation rate"
NON_OPERATED_ROTATION_RATE = "Non-operated rotation rate"
MOCAP_DURATION = "Mocap duration (hours)"


def create_demographics_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of demographics for each participant.

    Returns a DataFrame with the following columns:
        - participant: The participant's identifier.
        - filename: The participant's filename.
        - Age: The participant's age.
        - Sex: The participant's sex.
        - Handedness: The participant's handedness (dominant arm).
        - RTSA side: The side of the participant's RTSA (if any).
        - RTSA On Dominant Arm: True if RTSA is on the dominant arm, False if not, None if no RTSA.
        - Start Time: The start time of the participant's motion capture recording.
        - Mocap duration (hours): Total raw mocap recording duration.
    """

    rows = []
    for i, participant in enumerate(data):
        vals = {
            'participant': i,
            'filename': participant.filename,
            'Age': participant.age,
            'Sex': participant.sex,
            'Handedness': participant.dominant_arm,
            'RTSA side': participant.rtsa_side,
            'RTSA On Dominant Arm': participant.rtsa_on_dominant_arm,
            'Start Time': participant.start_time,
            MOCAP_DURATION: participant.mocap_duration,
        }
        rows.append(vals)

    return pd.DataFrame(
        rows,
        columns=[
            "participant",
            "filename",
            "Age",
            "Sex",
            "Handedness",
            "RTSA side",
            "RTSA On Dominant Arm",
            "Start Time",
            MOCAP_DURATION,
        ],
    )


def _create_analysis_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """
    Create a long-format dataframe for analysis, with each row representing a 
    single arm for a single participant. The table will have the following 
    columns:
        - participant: The participant's identifier.
        - arm: The arm (operated or non-operated).
        - rotation rate: The rotation rate for the arm.
        - age: The participant's age.
        - sex: The participant's sex.
        - dominant arm: Whether or not this is the participant's dominant arm (boolean).

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
    Returns:
        pd.DataFrame: Long-format DataFrame for analysis.
    """

    rows = []
    for i, participant in enumerate(data):
        for arm_type, arm in zip(
            ["operated", "non-operated"],
            [participant.operated[0], participant.non_operated[0]],
        ):
            vals = {
                'participant': i,
                'arm': arm_type,
                'rotation rate': arm.humerothoracic.rotation_rate,
                'age': participant.age,
                'sex': participant.sex,
                'dominant arm': (
                    True if participant.dominant_arm == arm_type else False
                ),
            }
            rows.append(vals)

    return pd.DataFrame(
        rows,
        columns=[
            "participant",
            "arm",
            "rotation rate",
            "age",
            "sex",
            "dominant arm",
        ],
    )


def create_rotation_data_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of cumulative rotation and rate for each participant.

    Returns a DataFrame with the following columns:
        - participant: The participant's identifier.
        - filename: The participant's filename.
        - Operated cumulative rotation: Cumulative rotation for the operated arm.
        - Non-operated cumulative rotation: Cumulative rotation for the non-operated arm.
        - Operated rotation rate: Rotation rate for the operated arm.
        - Non-operated rotation rate: Rotation rate for the non-operated arm.
        - Mocap duration (hours): Total raw mocap recording duration.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
    Returns:
        pd.DataFrame: DataFrame of cumulative totals for each participant.
    """

    rows = []
    for i, participant in enumerate(data):
        vals = {
            'participant': i,
            'filename': participant.filename,
            OPERATED_CUMULATIVE_ROTATION: participant.operated[0].humerothoracic.trace_total,
            NON_OPERATED_CUMULATIVE_ROTATION: participant.non_operated[0].humerothoracic.trace_total,
            OPERATED_ROTATION_RATE: participant.operated[0].humerothoracic.rotation_rate,
            NON_OPERATED_ROTATION_RATE: participant.non_operated[0].humerothoracic.rotation_rate,
            MOCAP_DURATION: participant.mocap_duration,
        }
        rows.append(vals)

    return pd.DataFrame(
        rows,
        columns=[
            "participant",
            "filename",
            OPERATED_CUMULATIVE_ROTATION,
            NON_OPERATED_CUMULATIVE_ROTATION,
            OPERATED_ROTATION_RATE,
            NON_OPERATED_ROTATION_RATE,
            MOCAP_DURATION,
        ],
    )


def _create_ttest_table(
    rotation_data: pd.DataFrame,
    operated_column: str,
    non_operated_column: str,
) -> pd.DataFrame:
    """Calculate paired summary statistics for one rotation metric."""
    t_stat, p_value = stats.ttest_rel(
        rotation_data[operated_column],
        rotation_data[non_operated_column],
    )

    return pd.DataFrame({
        "Arm": ["Operated", "Non-operated"],
        "n": [
            rotation_data[operated_column].shape[0],
            rotation_data[non_operated_column].shape[0],
        ],
        "Mean": [
            rotation_data[operated_column].mean(),
            rotation_data[non_operated_column].mean(),
        ],
        "Std": [
            rotation_data[operated_column].std(),
            rotation_data[non_operated_column].std(),
        ],
        "CofV": [
            rotation_data[operated_column].std(
            ) / rotation_data[operated_column].mean(),
            rotation_data[non_operated_column].std(
            ) / rotation_data[non_operated_column].mean(),
        ],
        "t-statistic": [t_stat, t_stat],
        "p-value": [p_value, p_value],
    })


def run_statistics(
    data: list[ParticipantDetails],
    out_path: Path = STATISTICS_OUTPUT_PATH
):
    demographics_data = create_demographics_dataframe(data)
    rotation_data = create_rotation_data_dataframe(data)
    cumulative_rotation_ttest = _create_ttest_table(
        rotation_data,
        OPERATED_CUMULATIVE_ROTATION,
        NON_OPERATED_CUMULATIVE_ROTATION,
    )
    rotation_rate_ttest = _create_ttest_table(
        rotation_data,
        OPERATED_ROTATION_RATE,
        NON_OPERATED_ROTATION_RATE,
    )

    # run t-test to see if rotation rate differed between sexes (expect sex values to be either 'M' or 'F')

    # run t-test to see if rotation rate differed between rtsa_on_dominant_arm values (expect rtsa_on_dominant_arm values to be either True or False)

    # Save the tables to an Excel file
    with pd.ExcelWriter(out_path) as writer:
        demographics_data.to_excel(
            writer,
            sheet_name="demographics",
            index=False
        )
        rotation_data.to_excel(
            writer,
            sheet_name="rotation_data_by_arm",
            index=False
        )
        cumulative_rotation_ttest.to_excel(
            writer,
            sheet_name="cumulative_rotation_stats",
            index=False,
        )
        rotation_rate_ttest.to_excel(
            writer,
            sheet_name="rotation_rate_stats",
            index=False,
        )


# -------------------------------------------------------------------
# main() runner to re-plot all figures if this module is run directly
# -------------------------------------------------------------------
def main():
    """Re-run all statistics if this module is run directly."""
    # Load data from pickle
    data = pd.read_pickle(Path(RESULTS_PICKLE_PATH))

    # Run all statistics
    run_statistics(data)


if __name__ == "__main__":
    main()
