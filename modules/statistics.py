"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

"""
from pathlib import Path

import pandas as pd
from scipy import stats
from config import CUMULATIVE_MOTION_STATISTICS_PATH
from schema import ParticipantDetails

# Constants for column names in the statistics DataFrame
OPERATED_CUMULATIVE_ROTATION = "Operated cumulative rotation"
NON_OPERATED_CUMULATIVE_ROTATION = "Non-operated cumulative rotation"
OPERATED_ROTATION_RATE = "Operated rotation rate"
NON_OPERATED_ROTATION_RATE = "Non-operated rotation rate"


def get_only_one_sided_participants(
    data: list[ParticipantDetails]
) -> list[ParticipantDetails]:
    """Return a list of ParticipantDetails objects with exactly one arm with RTSA and the other arm with no RTSA or TSA.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.

    Returns:
        list[ParticipantDetails]: List of ParticipantDetails objects.
    """
    single_arm_participants = [
        participant for participant in data if (
            participant.rtsa_side is not None
            and participant.rtsa_side != 'both'
            and participant.tsa_side is None
        )
    ]

    return single_arm_participants


def create_rotation_data_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of cumulative rotation and rate for each participant.

    Returns a DataFrame with the following columns:
        - participant: The participant's filename identifier.
        - Operated cumulative rotation: Cumulative rotation for the operated arm.
        - Non-operated cumulative rotation: Cumulative rotation for the non-operated arm.
        - Operated rotation rate: Rotation rate for the operated arm.
        - Non-operated rotation rate: Rotation rate for the non-operated arm.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
    Returns:
        pd.DataFrame: DataFrame of cumulative totals for each participant.
    """

    one_sided_participants = get_only_one_sided_participants(data)

    rows = []
    for participant in one_sided_participants:
        vals = {
            'participant': participant.filename,
            OPERATED_CUMULATIVE_ROTATION: participant.operated[0].humerothoracic.trace_total,
            NON_OPERATED_CUMULATIVE_ROTATION: participant.non_operated[0].humerothoracic.trace_total,
            OPERATED_ROTATION_RATE: participant.operated[0].humerothoracic.rotation_rate,
            NON_OPERATED_ROTATION_RATE: participant.non_operated[0].humerothoracic.rotation_rate,
        }
        rows.append(vals)

    return pd.DataFrame(
        rows,
        columns=[
            "participant",
            OPERATED_CUMULATIVE_ROTATION,
            NON_OPERATED_CUMULATIVE_ROTATION,
            OPERATED_ROTATION_RATE,
            NON_OPERATED_ROTATION_RATE,
        ],
    )


def _create_summary_table(
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
        "t-statistic": [t_stat, t_stat],
        "p-value": [p_value, p_value],
    })


def run_statistics(
    data: list[ParticipantDetails],
    out_path: Path = CUMULATIVE_MOTION_STATISTICS_PATH
):
    rotation_data = create_rotation_data_dataframe(data)
    cumulative_summary = _create_summary_table(
        rotation_data,
        OPERATED_CUMULATIVE_ROTATION,
        NON_OPERATED_CUMULATIVE_ROTATION,
    )
    rotation_rate_summary = _create_summary_table(
        rotation_data,
        OPERATED_ROTATION_RATE,
        NON_OPERATED_ROTATION_RATE,
    )

    # Save the tables to an Excel file
    with pd.ExcelWriter(out_path) as writer:
        rotation_data.to_excel(
            writer, sheet_name="rotation_data_by_arm", index=False)
        cumulative_summary.to_excel(
            writer,
            sheet_name="cumulative_rotation_summary",
            index=False,
        )
        rotation_rate_summary.to_excel(
            writer,
            sheet_name="rotation_rate_summary",
            index=False,
        )
