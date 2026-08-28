"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

"""
from pathlib import Path

import pandas as pd
from scipy import stats
from typing import Literal

from config import CUMULATIVE_MOTION_STATISTICS_PATH
from schema import ParticipantDetails


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


def create_cumulative_totals_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of cumulative totals for each participant.

    Returns a DataFrame with the following columns:
        - participant: The participant's filename identifier.
        - Operated: The cumulative total rotation for the operated arm.
        - Non-operated: The cumulative total rotation for the non-operated arm.

    Args:
        data (list[ParticipantDetails]): List of ParticipantDetails objects.
    Returns:
        pd.DataFrame: DataFrame of cumulative totals for each participant.
    """

    one_sided_participants = get_only_one_sided_participants(data)

    rows = []
    for idx, participant in enumerate(one_sided_participants):
        vals = {
            'participant': participant.filename,
            'Operated': participant.operated[0].humerothoracic.trace_total,
            'Non-operated': participant.non_operated[0].humerothoracic.trace_total
        }
        rows.append(vals)

    cumulative_totals = pd.DataFrame(rows)
    return cumulative_totals


def create_summary_table(cumulative_totals: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate summary statistics table for operated and non-operated groups.

    Parameters:
        cumulative_totals (pd.DataFrame): Dictionary containing cumulative totals for each group.

    Returns:
        pd.DataFrame: A DataFrame containing the summary statistics.
    """

    # run t-test
    t_stat, p_value = stats.ttest_rel(
        cumulative_totals['Operated'],
        cumulative_totals['Non-operated']
    )

    summary_table = pd.DataFrame({
        "Arm": ["Operated", "Non-operated"],
        "n": [cumulative_totals["Operated"].shape[0], cumulative_totals["Non-operated"].shape[0]],
        "Mean": [cumulative_totals["Operated"].mean(), cumulative_totals["Non-operated"].mean()],
        "Std": [cumulative_totals["Operated"].std(), cumulative_totals["Non-operated"].std()],
        "t-statistic": [t_stat, t_stat],
        "p-value": [p_value, p_value]
    })

    return summary_table


def run_statistics(
    data: list[ParticipantDetails],
    out_path: Path = CUMULATIVE_MOTION_STATISTICS_PATH
):
    cumulative_totals = create_cumulative_totals_dataframe(data)
    summary_table = create_summary_table(cumulative_totals)

    # Save the tables to an Excel file
    with pd.ExcelWriter(out_path) as writer:
        cumulative_totals.to_excel(writer, sheet_name="cumulative_rotation_by_arm", index=False)
        summary_table.to_excel(writer, sheet_name="summary_table_for_cumulative_rotation", index=False)
