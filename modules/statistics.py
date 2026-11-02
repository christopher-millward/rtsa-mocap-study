"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

Objective
---------
Describe how much motion is observed in RTSA shoulders: the distribution of
rotation rate in

1. operated and non-operated shoulders,
2. operated shoulders of male and female participants, and
3. operated shoulders where the RTSA is on the dominant and on the
   non-dominant arm.

The analysis is descriptive. It reports the distributions; it does not try
to explain why rotation rate differs across participants. A t-test is
reported for each comparison as a simple addition.

Statistical approach
--------------------
* Each group: n, mean with 95% CI, SD, coefficient of variation, median,
  quartiles, range, skewness and a Shapiro-Wilk p-value.
* Operated vs non-operated: every participant contributes one measurement
  from each arm, so the arms are compared with a paired-samples t-test
  (`scipy.stats.ttest_rel`) on each participant's operated minus
  non-operated difference, with n - 1 degrees of freedom. A Wilcoxon
  signed-rank test is reported alongside as a sensitivity check.
* Male vs female, and dominant vs non-dominant RTSA: every participant has
  one operated arm, so these are independent groups. They are compared with
  Welch's two-sample t-test (`scipy.stats.ttest_ind` with
  ``equal_var=False``, which does not assume equal variances), with a
  Mann-Whitney U test as the sensitivity check. Its degrees of freedom come
  from the Welch-Satterthwaite approximation, so they are usually not a
  whole number. The SD, skewness and
  Shapiro-Wilk p of per-participant differences are not defined for
  independent groups, so they are marked as not applicable; each group's own
  skewness and Shapiro-Wilk p are in its distribution table.

No p-value adjustment is applied. All tests are two-sided with alpha = 0.05.
"""

import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

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
SEX = "Sex"

# -------------------------------------------------------------------
# Constants for the analysis DataFrame (one row per participant)
# -------------------------------------------------------------------
PARTICIPANT = "participant"
FILENAME = "filename"
OPERATED = "operated"
NON_OPERATED = "non-operated"
SEX_CODE = "sex"
DOMINANT_ARM_SURGERY = "rtsa_on_dominant_arm"

VALID_SEX_CODES = ("F", "M")

# -------------------------------------------------------------------
# Constants for statistical settings and result tables
# -------------------------------------------------------------------
ALPHA = 0.05
MIN_PARTICIPANTS_FOR_TEST = 3

STATISTIC = "Statistic"
CI_LOWER = f"{1 - ALPHA:.0%} CI lower"
CI_UPPER = f"{1 - ALPHA:.0%} CI upper"
NOT_APPLICABLE = "N/A (independent groups)"

# Continuous demographic variables: (row label, demographics column,
# decimal places, is a clock time).
DEMOGRAPHIC_CONTINUOUS_VARIABLES: list[tuple[str, str, int, bool]] = [
    ("Age (years)", "Age", 1, False),
    ("Recording start time (hh:mm)", "Start Time", 0, True),
    ("Recording duration (hours)", MOCAP_DURATION, 1, False),
]

# Categorical demographic variables shown as n (%): (row label,
# statistic label, demographics column, level counted).
DEMOGRAPHIC_COUNT_VARIABLES: list[tuple[str, str, str, object]] = [
    ("RTSA side, n (%)", "Left", "RTSA side", "left"),
    ("RTSA side, n (%)", "Right", "RTSA side", "right"),
    ("RTSA on dominant arm, n (%)", "Yes", "RTSA On Dominant Arm", True),
]


# -------------------------------------------------------------------
# Helper functions: building DataFrames
# -------------------------------------------------------------------

def _create_demographics_dataframe(
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


def create_rotation_data_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of cumulative rotation and rate for each participant.

    Args:
        data: List of ParticipantDetails objects.

    Returns:
        DataFrame with one row per participant and the following columns:
            - participant: The participant's identifier.
            - filename: The participant's filename.
            - Operated cumulative rotation: Cumulative rotation for the
              operated arm.
            - Non-operated cumulative rotation: Cumulative rotation for the
              non-operated arm.
            - Operated rotation rate: Rotation rate for the operated arm.
            - Non-operated rotation rate: Rotation rate for the non-operated
              arm.
            - Mocap duration (hours): Total raw mocap recording duration.
    """

    rows = []
    for i, participant in enumerate(data):
        operated = participant.operated[0].humerothoracic
        non_operated = participant.non_operated[0].humerothoracic
        vals = {
            'participant': i,
            'filename': participant.filename,
            OPERATED_CUMULATIVE_ROTATION: operated.trace_total,
            NON_OPERATED_CUMULATIVE_ROTATION: non_operated.trace_total,
            OPERATED_ROTATION_RATE: operated.rotation_rate,
            NON_OPERATED_ROTATION_RATE: non_operated.rotation_rate,
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


def _create_analysis_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a DataFrame of rotation rates with one row per participant.

    Participant identifiers match those used in the demographics and
    rotation-data tables (the participant's position in ``data``).

    Args:
        data: List of ParticipantDetails objects.

    Returns:
        DataFrame with the columns `participant`, `filename`, `operated`
        and `non-operated` (the rotation rate of each arm), `sex` and
        `rtsa_on_dominant_arm`. Missing or non-finite rotation rates are
        stored as NaN. Sex and dominant-arm surgery are stored exactly as
        recorded; they are checked by `_validate_analysis_data`.
    """
    rows = []
    for participant_id, participant in enumerate(data):
        rows.append({
            PARTICIPANT: participant_id,
            FILENAME: participant.filename,
            OPERATED: _to_finite_float(
                participant.operated[0].humerothoracic.rotation_rate
            ),
            NON_OPERATED: _to_finite_float(
                participant.non_operated[0].humerothoracic.rotation_rate
            ),
            SEX_CODE: participant.sex,
            DOMINANT_ARM_SURGERY: participant.rtsa_on_dominant_arm,
        })

    return pd.DataFrame(rows)


def _to_finite_float(value: object) -> float:
    """Convert a measurement to float, returning NaN if missing or infinite.

    Raises:
        TypeError or ValueError: If the value cannot be read as a number,
            because that indicates a data problem rather than a missing
            measurement.
    """
    if value is None:
        return np.nan

    # intentionally let this raise if value is not a number
    number = float(value)  # type:ignore

    return number if np.isfinite(number) else np.nan


# -------------------------------------------------------------------
# Helper functions: validation and complete-case selection
# -------------------------------------------------------------------

def _validate_analysis_data(analysis_data: pd.DataFrame) -> pd.DataFrame:
    """Check the analysis data before any test is run.

    * Participants sharing a filename trigger a warning, because they may be
      the same person recorded twice (which would break independence).
    * Sex codes other than "M"/"F" and dominant-arm-surgery values other
      than True/False are set to missing, with a warning. They are never
      recoded, and the affected participants are only excluded from the
      comparison that needs that variable.

    Args:
        analysis_data: Output of `_create_analysis_dataframe`.

    Returns:
        A validated copy of `analysis_data`.
    """
    repeated_files = analysis_data[FILENAME].duplicated(keep=False)
    if repeated_files.any():
        warnings.warn(
            "These participants share a filename and may be the same person: "
            f"{analysis_data.loc[repeated_files, PARTICIPANT].tolist()}"
        )

    validated = analysis_data.copy()
    validated[SEX_CODE] = _set_invalid_values_to_missing(
        validated,
        column=SEX_CODE,
        is_valid=lambda value: value in VALID_SEX_CODES,
        expected=f"one of {VALID_SEX_CODES}",
    )
    validated[DOMINANT_ARM_SURGERY] = _set_invalid_values_to_missing(
        validated,
        column=DOMINANT_ARM_SURGERY,
        is_valid=lambda value: isinstance(value, (bool, np.bool_)),
        expected="True or False",
    )
    return validated


def _set_invalid_values_to_missing(
    analysis_data: pd.DataFrame,
    column: str,
    is_valid: Callable[[object], bool],
    expected: str,
) -> pd.Series:
    """Return ``column`` with invalid values replaced by NaN, warning if any.

    Args:
        analysis_data: Analysis data (one row per participant).
        column: Name of the column to check.
        is_valid: Function returning True for an acceptable value.
        expected: Description of acceptable values, used in the warning.

    Returns:
        A copy of the column in which invalid values are NaN.
    """
    values = analysis_data[column]
    invalid = ~values.map(is_valid).astype(bool)
    if invalid.any():
        bad_values = sorted({str(value) for value in values[invalid]})
        bad_participants = analysis_data.loc[invalid, PARTICIPANT].tolist()
        warnings.warn(
            f"'{column}' has unexpected values {bad_values} (expected "
            f"{expected}) for participants {bad_participants}. They are "
            "treated as missing and excluded only from the comparison that "
            f"uses '{column}'."
        )
    return values.where(~invalid)


def _select_complete_cases(
    analysis_data: pd.DataFrame,
    columns: list[str],
    comparison_name: str,
) -> pd.DataFrame:
    """Keep participants with a value in every one of ``columns``.

    Each comparison calls this separately, so a participant who is missing
    one value is only dropped from the comparisons that use it. Every
    exclusion is raised as a warning.

    Args:
        analysis_data: Validated analysis data.
        columns: Columns that must be present.
        comparison_name: Name used in exclusion messages.

    Returns:
        The complete-case subset.
    """
    missing = analysis_data[columns].isna()
    for row_index in missing.index[missing.any(axis=1)]:
        missing_columns = missing.columns[missing.loc[row_index]].tolist()
        warnings.warn(
            f"{comparison_name}: participant "
            f"{analysis_data.at[row_index, PARTICIPANT]} excluded (missing "
            f"{', '.join(missing_columns)})."
        )
    return analysis_data[~missing.any(axis=1)].copy()


def _check_enough_data(
    group_sizes: dict[str, int],
    comparison_name: str,
) -> None:
    """Raise an error if any group is too small to describe and test.

    Raises:
        ValueError: If a group has fewer than MIN_PARTICIPANTS_FOR_TEST
            participants.
    """
    too_small = {
        label: n for label, n in group_sizes.items()
        if n < MIN_PARTICIPANTS_FOR_TEST
    }
    if too_small:
        raise ValueError(
            f"{comparison_name}: groups {too_small} have fewer than "
            f"{MIN_PARTICIPANTS_FOR_TEST} participants, which is too few to "
            "test."
        )


# -------------------------------------------------------------------
# Helper functions: distribution statistics
# -------------------------------------------------------------------

def _shapiro_p_value(values: pd.Series) -> float:
    """Return the Shapiro-Wilk p-value, or NaN if fewer than 3 values.

    This is reported as a diagnostic only. It is never used to choose a
    test, because with small samples it has little power and with large
    samples it flags trivial departures.
    """
    if len(values) < 3:
        return np.nan
    _, p_value = stats.shapiro(values)
    return p_value


def _skewness(values: pd.Series) -> float:
    """Return the sample skewness, or NaN if fewer than 3 values."""
    if len(values) < 3:
        return np.nan
    return stats.skew(values)


def _describe_distribution(values: pd.Series) -> dict[str, float]:
    """Summarise the distribution of one set of values.

    The mean's confidence interval is a t-interval
    (`scipy.stats.ttest_1samp`). The median, quartiles and range are
    reported alongside because rotation rates may be skewed. The
    coefficient of variation is the SD divided by the mean.

    Args:
        values: Values to summarise (missing values must already be removed).

    Returns:
        Mapping of statistic name to value. Statistics that need more data
        than is available are NaN.
    """
    n = len(values)
    mean_ci_low = mean_ci_high = np.nan
    if n >= 2:
        mean_ci = stats.ttest_1samp(values, 0).confidence_interval(
            confidence_level=1 - ALPHA
        )
        mean_ci_low, mean_ci_high = mean_ci.low, mean_ci.high
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])

    return {
        "n": n,
        "Mean": values.mean(),
        f"Mean {CI_LOWER}": mean_ci_low,
        f"Mean {CI_UPPER}": mean_ci_high,
        "Std": values.std(),
        "CoV": values.std() / values.mean(),
        "Median": median,
        "Q1 (25th percentile)": q1,
        "Q3 (75th percentile)": q3,
        "Min": values.min(),
        "Max": values.max(),
        "Skewness": _skewness(values),
        "Shapiro-Wilk p": _shapiro_p_value(values),
    }


def _create_distribution_table(groups: dict[str, pd.Series]) -> pd.DataFrame:
    """Describe each group's distribution, one column per group.

    Args:
        groups: Mapping of group label to the values in that group.

    Returns:
        DataFrame with a "Statistic" column (see `_describe_distribution`)
        followed by one column per group.
    """
    return _statistics_table({
        label: _describe_distribution(values)
        for label, values in groups.items()
    })


def _statistics_table(
    columns: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame:
    """Lay out statistics with one row per statistic and one column each.

    Args:
        columns: Mapping of column label to {statistic name: value}. Every
            column must have the same statistics, in the same order.
    """
    return pd.DataFrame(columns).rename_axis(STATISTIC).reset_index()


# -------------------------------------------------------------------
# Helper functions: demographics summary
# -------------------------------------------------------------------

def _time_to_hours(clock_time: datetime) -> float:
    """Convert a `datetime` to decimal hours after midnight.

    Returns NaN for a missing value, so missing start times are left out
    of the summary rather than treated as midnight.
    """
    if clock_time is None or pd.isna(clock_time):
        return np.nan

    return (
        clock_time.hour
        + clock_time.minute / 60
        + clock_time.second / 3600
    )


def _format_clock(hours: float) -> str:
    """Format decimal hours after midnight as a 24-hour "hh:mm" string."""
    total_minutes = round(hours * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _summarize_continuous_variable(
    values: pd.Series,
    decimals: int,
    is_clock_time: bool,
) -> dict[str, str]:
    """Describe the distribution of one continuous variable as text.

    Uses `_describe_distribution` so these statistics are calculated the
    same way as in the rotation-rate distribution tables.

    Args:
        values: Values for one group of participants (may contain NaN).
        decimals: Decimal places for values in the variable's units.
        is_clock_time: True if `values` are hours after midnight. Clock
            values are shown as hh:mm and the SD in minutes.

    Returns:
        Mapping of statistic label to formatted cell text.
    """
    values = values.dropna()
    if values.empty:
        return {"n": "0"}
    summary = _describe_distribution(values)

    def format_value(value: float) -> str:
        if np.isnan(value):  # e.g. SD or skewness of a very small group
            return "-"
        if is_clock_time:
            return _format_clock(value)
        return f"{value:.{decimals}f}"

    if is_clock_time and not np.isnan(summary["Std"]):
        spread = f"{summary['Std'] * 60:.0f} min"
    else:
        spread = format_value(summary["Std"])
    skewness = summary["Skewness"]

    return {
        "n": f"{summary['n']}",
        "Mean ± SD": f"{format_value(summary['Mean'])} ± {spread}",
        "Median [IQR]": (
            f"{format_value(summary['Median'])} "
            f"[{format_value(summary['Q1 (25th percentile)'])}–"
            f"{format_value(summary['Q3 (75th percentile)'])}]"
        ),
        "Range (min–max)": (
            f"{format_value(summary['Min'])}–{format_value(summary['Max'])}"
        ),
        "Skewness": "–" if np.isnan(skewness) else f"{skewness:.2f}",
    }


def _summarize_count(values: pd.Series, level: object) -> str:
    """Return "n (%)" for one level of a categorical variable.

    The percentage uses participants with a recorded value as the
    denominator, so missing values do not count as "not this level".
    """
    recorded = values.dropna()
    count = int((recorded == level).sum())
    if recorded.empty:
        return "0 (-)"
    return f"{count} ({count / len(recorded):.0%})"


def _create_demographics_summary_table(
    demographics_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize participant characteristics by sex, ready for publication.

    The table follows the usual "Table 1" layout: one block of rows per
    characteristic and one column each for female, male and all
    participants, with the group size in the column header. Continuous
    variables (age, start time, duration) show n, mean ± SD, median [IQR],
    range and skewness. Categorical variables (RTSA side, RTSA on dominant
    arm) show n (%), because a mean, range or skewness of a yes/no
    variable carries no extra information.

    Every participant is in the "All" column, including anyone whose sex is
    not recorded as "F" or "M".

    Args:
        demographics_data: Output of `_create_demographics_dataframe`.

    Returns:
        DataFrame with columns "Characteristic", "Statistic" and one column
        per group. The characteristic name appears only on the first row of
        its block, for readability. All cells are formatted text.
    """
    groups = {
        "Female": demographics_data[demographics_data[SEX] == "F"],
        "Male": demographics_data[demographics_data[SEX] == "M"],
        "All": demographics_data,
    }
    group_columns = {
        name: f"{name} (n = {len(group)})" for name, group in groups.items()
    }

    rows = []
    for label, column, decimals, is_clock_time in (
        DEMOGRAPHIC_CONTINUOUS_VARIABLES
    ):
        summaries = {}
        for name, group in groups.items():
            values = group[column]
            if is_clock_time:
                values = values.map(_time_to_hours)
            summaries[name] = _summarize_continuous_variable(
                values, decimals, is_clock_time
            )

        for statistic in summaries["All"]:
            rows.append({
                "Characteristic": label,
                "Statistic": statistic,
                **{
                    group_columns[name]: summary.get(statistic, "-")
                    for name, summary in summaries.items()
                },
            })

    for label, statistic, column, level in DEMOGRAPHIC_COUNT_VARIABLES:
        rows.append({
            "Characteristic": label,
            "Statistic": statistic,
            **{
                group_columns[name]: _summarize_count(group[column], level)
                for name, group in groups.items()
            },
        })

    table = pd.DataFrame(rows)
    repeated_label = table["Characteristic"].eq(
        table["Characteristic"].shift()
    )
    table.loc[repeated_label, "Characteristic"] = ""
    return table


# -------------------------------------------------------------------
# Helper functions: comparisons of rotation rate
# -------------------------------------------------------------------

def _compare_operated_to_non_operated(
    analysis_data: pd.DataFrame,
) -> list[pd.DataFrame]:
    """Compare operated and non-operated arms with a paired t-test.

    The paired t-test analyses each participant's operated minus
    non-operated difference, so between-participant variation cancels out.
    It assumes the differences are approximately normal; with n around 20
    it is reasonably robust to moderate skew. A Wilcoxon signed-rank test is
    reported alongside as a sensitivity check (it tests a shift in the
    distribution of differences rather than the mean difference).

    Only participants with both arms measured are included, so the mean
    difference equals the difference between the two arms' means.

    Args:
        analysis_data: Validated analysis data.

    Returns:
        Two tables: the distribution of rotation rate in each arm, and the
        statistics of the paired differences.
    """
    comparison_name = "Operated vs non-operated"
    pairs = _select_complete_cases(
        analysis_data, [OPERATED, NON_OPERATED], comparison_name
    )
    _check_enough_data({"pairs": len(pairs)}, comparison_name)

    differences = pairs[OPERATED] - pairs[NON_OPERATED]
    # scipy builds its result classes dynamically, so the linter cannot see
    # their `statistic`/`pvalue` attributes; unpacking the tuples avoids this.
    test = stats.ttest_rel(pairs[OPERATED], pairs[NON_OPERATED])
    t_statistic, p_value = test
    degrees_of_freedom = test.df  # n - 1 for a paired t-test
    confidence_interval = test.confidence_interval(confidence_level=1 - ALPHA)
    _, wilcoxon_p = stats.wilcoxon(differences)

    distribution_table = _create_distribution_table(
        {"Operated": pairs[OPERATED], "Non-operated": pairs[NON_OPERATED]}
    )
    difference_table = _statistics_table({
        "Operated - Non-operated": {
            "Test": "Paired-samples t-test",
            "n": len(differences),
            "Mean": differences.mean(),
            "Std": differences.std(),
            CI_LOWER: confidence_interval.low,
            CI_UPPER: confidence_interval.high,
            "t-statistic": t_statistic,
            "Degrees of freedom": degrees_of_freedom,
            "p-value": p_value,
            "Shapiro-Wilk p": _shapiro_p_value(differences),
            "Skewness": _skewness(differences),
            "Wilcoxon signed-rank p": wilcoxon_p,
        }
    })
    return [distribution_table, difference_table]


def _compare_operated_arm_groups(
    analysis_data: pd.DataFrame,
    column: str,
    levels: dict[str, object],
    comparison_name: str,
) -> list[pd.DataFrame]:
    """Compare operated-arm rotation rate between two independent groups.

    Each participant contributes one operated arm, so the two groups are
    independent and are compared with Welch's two-sample t-test, which does
    not assume equal variances (group sizes here are unequal). A
    Mann-Whitney U test is reported alongside as a sensitivity check; it is
    the independent-groups counterpart of the Wilcoxon signed-rank test.

    The SD, skewness and Shapiro-Wilk p of per-participant differences only
    exist for paired data, so they are marked as not applicable. The skewness
    and Shapiro-Wilk p of each group are in the distribution table.

    Args:
        analysis_data: Validated analysis data.
        column: Participant-level column that defines the groups.
        levels: Mapping of group label to the value of ``column`` for that
            group, e.g. ``{"Male": "M", "Female": "F"}``. The difference is
            the first group minus the second.
        comparison_name: Name used in warning and error messages.

    Returns:
        Two tables: the distribution of operated-arm rotation rate in each
        group, and the statistics of the difference between the groups.
    """
    cases = _select_complete_cases(
        analysis_data, [OPERATED, column], comparison_name
    )
    groups = {
        label: cases.loc[cases[column] == level, OPERATED]
        for label, level in levels.items()
    }
    _check_enough_data(
        {label: len(values) for label, values in groups.items()},
        comparison_name,
    )

    (first_label, first), (second_label, second) = groups.items()
    # scipy builds its result classes dynamically, so the linter cannot see
    # their `statistic`/`pvalue` attributes; unpacking the tuples avoids this.
    test = stats.ttest_ind(first, second, equal_var=False)
    t_statistic, p_value = test
    # `df` is not part of the tuple, so it must be read as an attribute and
    # the linter error silenced. It is the Welch-Satterthwaite df, which is
    # usually fractional.
    degrees_of_freedom = test.df  # type: ignore
    confidence_interval = test.confidence_interval(confidence_level=1 - ALPHA)
    _, mann_whitney_p = stats.mannwhitneyu(first, second)

    distribution_table = _create_distribution_table(groups)
    difference_table = _statistics_table({
        f"{first_label} - {second_label}": {
            "Test": "Welch's two-sample t-test",
            "n": len(first) + len(second),
            "Mean": first.mean() - second.mean(),
            "Std": NOT_APPLICABLE,
            CI_LOWER: confidence_interval.low,
            CI_UPPER: confidence_interval.high,
            "t-statistic": t_statistic,
            "Degrees of freedom": degrees_of_freedom,
            "p-value": p_value,
            "Shapiro-Wilk p": NOT_APPLICABLE,
            "Skewness": NOT_APPLICABLE,
            "Mann-Whitney U p": mann_whitney_p,
        }
    })
    return [distribution_table, difference_table]


# -------------------------------------------------------------------
# Helper functions: output
# -------------------------------------------------------------------

def _save_tables_to_excel(
    sheets: dict[str, list[pd.DataFrame]],
    out_path: Path,
) -> None:
    """Write each sheet's tables one below another, in the order given.

    Tables on the same sheet are separated by a blank row.
    """
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name, tables in sheets.items():
            start_row = 0
            for table in tables:
                table.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=start_row,
                    index=False,
                )
                start_row += len(table) + 2  # header row + blank row


# -------------------------------------------------------------------
# Main statistics pipeline
# -------------------------------------------------------------------

def run_statistics(
    data: list[ParticipantDetails],
    out_path: Path = STATISTICS_OUTPUT_PATH
) -> None:
    """Run all statistical analyses and save the results to Excel.

    Args:
        data: List of ParticipantDetails objects, one per participant.
        out_path: Path of the Excel workbook to write.
    """
    # 1. Descriptive tables (one row per participant)
    demographics_data = _create_demographics_dataframe(data)
    rotation_data = create_rotation_data_dataframe(data)

    # 2. Demographics summary table
    demographics_summary = _create_demographics_summary_table(
        demographics_data
    )

    # 3. Rotation rates for analysis (one row per participant), validated
    analysis_data = _validate_analysis_data(_create_analysis_dataframe(data))

    # 4. Rotation rate distributions and t-tests for each comparison
    operated_vs_non_operated = _compare_operated_to_non_operated(
        analysis_data
    )
    male_vs_female = _compare_operated_arm_groups(
        analysis_data,
        column=SEX_CODE,
        levels={"Male": "M", "Female": "F"},
        comparison_name="Male vs female operated arms",
    )
    dominant_vs_non_dominant = _compare_operated_arm_groups(
        analysis_data,
        column=DOMINANT_ARM_SURGERY,
        levels={"Dominant": True, "Non-dominant": False},
        comparison_name="Dominant vs non-dominant operated arms",
    )

    # 5. Save the tables to an Excel file
    _save_tables_to_excel(
        {
            "raw_demographics": [demographics_data],
            "demographics_summary": [demographics_summary],
            "rotation_data_by_arm": [rotation_data],
            "operated_vs_non_operated": operated_vs_non_operated,
            "male_vs_female": male_vs_female,
            "dominant_vs_non_dominant": dominant_vs_non_dominant,
        },
        out_path,
    )


# -------------------------------------------------------------------
# main() runner to re-run all statistics if this module is run directly
# -------------------------------------------------------------------
def main() -> None:
    """Re-run all statistics if this module is run directly."""
    # Load data from pickle
    data = pd.read_pickle(Path(RESULTS_PICKLE_PATH))

    # Run all statistics
    run_statistics(data)


if __name__ == "__main__":
    main()
