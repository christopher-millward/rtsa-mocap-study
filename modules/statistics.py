"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

Objectives
----------
Primary (descriptive): describe the distribution of rotation rate in the
operated (RTSA) arm.

Secondary (inferential):
1. Is there a difference between operated and non-operated rotation rates?
2. Is there a difference after controlling for sex, and is part of the
   operated vs non-operated difference explained by sex?
3. ... after controlling for whether the RTSA was on the dominant arm?
4. ... after controlling for both sex and dominant-arm surgery?

Data structure
--------------
Every participant contributes one measurement from each arm, so the two
measurements from one participant are not independent. Sex and "RTSA on
dominant arm" are participant-level characteristics: they are constant
across a participant's two arms.

Statistical approach
--------------------
* Primary: n, mean with 95% CI, SD, median, quartiles, range, skewness and
  a Shapiro-Wilk p-value for the operated arm (with the non-operated arm and
  the paired difference for context), overall and within each subgroup.
* Secondary Q1: paired-samples t-test (`scipy.stats.ttest_rel`). Pairs
  are matched on participant identity.
* Secondary Q2-Q4: linear regression (`statsmodels` OLS) of each
  participant's paired difference (operated minus non-operated):

      Model A: difference ~ 1                       (= the paired t-test)
      Model B: difference ~ sex
      Model C: difference ~ dominant-arm surgery
      Model D: difference ~ sex + dominant-arm surgery

  Covariates are effect-coded as -0.5/+0.5, so:

  - The intercept is the operated vs non-operated difference, adjusted for
    the covariates (averaged with equal weight over the covariate groups).
  - Each covariate coefficient is how much the operated vs non-operated
    difference changes between that covariate's groups (e.g. the
    difference in men minus the difference in women). A non-zero
    coefficient means part of the arm difference is explained by that
    covariate.

  Why regress the differences? With exactly two arms per participant, this
  gives the same estimates as the linear mixed model
  ``rotation_rate ~ arm * covariates + (1 | participant)``: the
  participant's own baseline cancels out when the two arms are subtracted.
  The regression is simpler and gives exact small-sample t-tests with
  n - p degrees of freedom, whereas ``statsmodels`` mixed models only give
  large-sample Wald z-tests.

  Interpreting dominant-arm surgery: when the RTSA is on the dominant arm,
  the difference is (surgery effect + handedness effect); when it is on the
  non-dominant arm, it is (surgery effect - handedness effect). In Model C
  the intercept is therefore the surgery effect with handedness removed
  (assuming the handedness effect is the same size in both groups), and the
  coefficient is twice the dominant arm's advantage.

  Additive models only: no sex x dominant-arm-surgery term is included.

Multiple comparisons
--------------------
No p-value adjustment is applied. The inferential analyses are secondary,
and Models A-D are nested views of one question rather than independent
hypotheses. Results should be read as exploratory.

Age is intentionally not analyzed. All tests use alpha = 0.05.
"""
# from __future__ import annotations
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
SEX = "Sex"

# -------------------------------------------------------------------
# Constants for the long-format analysis DataFrame (one row per arm).
# Names are snake_case so they can be used directly in model formulas.
# -------------------------------------------------------------------
PARTICIPANT = "participant"
FILENAME = "filename"
ARM = "arm"
ROTATION_RATE = "rotation_rate"
CUMULATIVE_ROTATION = "cumulative_rotation"
SEX_CODE = "sex"
DOMINANT_ARM_SURGERY = "rtsa_on_dominant_arm"

OPERATED = "operated"
NON_OPERATED = "non-operated"
VALID_SEX_CODES = ("F", "M")

# Column added when the long data are reshaped to one row per participant.
DIFFERENCE = "difference"

# -------------------------------------------------------------------
# Constants for statistical settings and result tables
# -------------------------------------------------------------------
ALPHA = 0.05
MIN_PARTICIPANTS_FOR_TEST = 3
MIN_PARTICIPANTS_PER_COVARIATE_LEVEL = 5

CI_LOWER = f"{1 - ALPHA:.0%} CI lower"
CI_UPPER = f"{1 - ALPHA:.0%} CI upper"
NOTES = "Notes"
NO_NOTES = "No exclusions or warnings"

# Covariates included in each paired-difference model.
DIFFERENCE_MODELS: dict[str, list[str]] = {
    "A: no covariates": [],
    "B: sex": [SEX_CODE],
    "C: dominant-arm surgery": [DOMINANT_ARM_SURGERY],
    "D: sex + dominant-arm surgery": [SEX_CODE, DOMINANT_ARM_SURGERY],
}

# Effect coding (-0.5 / +0.5) of each covariate, and the name of the coded
# column used in the model formulas.
EFFECT_CODES: dict[str, tuple[str, dict[object, float]]] = {
    SEX_CODE: ("sex_m_vs_f", {"F": -0.5, "M": 0.5}),
    DOMINANT_ARM_SURGERY: (
        "dominant_vs_nondominant_surgery", {False: -0.5, True: 0.5}
    ),
}

# Human-readable labels for the coefficient names produced by the formulas.
TERM_LABELS = {
    "Intercept": "Operated - non-operated difference (adjusted)",
    "sex_m_vs_f": "Change in difference: M vs F",
    "dominant_vs_nondominant_surgery": (
        "Change in difference: RTSA on dominant vs non-dominant arm"
    ),
}

# Subgroups for the descriptive tables: (label, column, level). A column of
# None means all participants.
DESCRIPTIVE_SUBGROUPS: list[tuple[str, str | None, object]] = [
    ("All participants", None, None),
    ("Sex = F", SEX_CODE, "F"),
    ("Sex = M", SEX_CODE, "M"),
    ("RTSA on dominant arm = True", DOMINANT_ARM_SURGERY, True),
    ("RTSA on dominant arm = False", DOMINANT_ARM_SURGERY, False),
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
    """Create a long-format DataFrame with one row per arm per participant.

    Participant identifiers match those used in the demographics and
    rotation-data tables (the participant's position in ``data``).

    Args:
        data: List of ParticipantDetails objects.

    Returns:
        DataFrame with the columns `participant`, `filename`, `arm`
        ("operated" or "non-operated"), `rotation_rate`,
        `cumulative_rotation`, `sex` and `rtsa_on_dominant_arm`.
        Missing or non-finite outcome values are stored as NaN. Sex and
        dominant-arm surgery are stored exactly as recorded; they are checked
        by `_validate_analysis_data`.
    """
    rows = []
    for participant_id, participant in enumerate(data):
        arms = {
            OPERATED: participant.operated[0],
            NON_OPERATED: participant.non_operated[0],
        }
        for arm_status, arm in arms.items():
            rows.append({
                PARTICIPANT: participant_id,
                FILENAME: participant.filename,
                ARM: arm_status,
                ROTATION_RATE: _to_finite_float(
                    arm.humerothoracic.rotation_rate
                ),
                CUMULATIVE_ROTATION: _to_finite_float(
                    arm.humerothoracic.trace_total
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
    """Check the long-format analysis data before any test is run.

    * Duplicate participant/arm rows raise an error, because the pairing of
      operated and non-operated arms would be ambiguous.
    * Participants sharing a filename trigger a warning, because they may be
      the same person recorded twice (which would break independence).
    * Sex codes other than "M"/"F" and dominant-arm-surgery values other
      than True/False are set to missing, with a warning. They are never
      recoded, and the affected participants are only excluded from the
      analyses that need that variable.

    Args:
        analysis_data: Output of `_create_analysis_dataframe`.

    Returns:
        A validated copy of `analysis_data`.

    Raises:
        ValueError: If a participant has more than one row for the same arm.
    """
    duplicate_rows = analysis_data.duplicated([PARTICIPANT, ARM], keep=False)
    if duplicate_rows.any():
        duplicated_ids = sorted(
            analysis_data.loc[duplicate_rows, PARTICIPANT].unique().tolist()
        )
        raise ValueError(
            f"Participants {duplicated_ids} have more than one measurement "
            "for the same arm, so arms cannot be paired reliably."
        )

    participant_files = analysis_data.drop_duplicates(PARTICIPANT)
    repeated_files = participant_files[FILENAME].duplicated(keep=False)
    if repeated_files.any():
        warnings.warn(
            "These participants share a filename and may be the same person: "
            f"{participant_files.loc[repeated_files, PARTICIPANT].tolist()}"
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
        analysis_data: Long-format analysis data.
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
        bad_participants = sorted(
            analysis_data.loc[invalid, PARTICIPANT].unique().tolist()
        )
        warnings.warn(
            f"'{column}' has unexpected values {bad_values} (expected "
            f"{expected}) for participants {bad_participants}. They are "
            "treated as missing and excluded only from analyses that use "
            f"'{column}'."
        )
    return values.where(~invalid)


def _select_complete_cases(
    analysis_data: pd.DataFrame,
    outcome: str,
    covariates: list[str],
    analysis_name: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep participants with both arms measured and all covariates present.

    Each analysis calls this separately, so a participant who is missing one
    covariate is only dropped from analyses that use that covariate.

    Args:
        analysis_data: Validated long-format analysis data.
        outcome: Outcome column that must be present for both arms.
        covariates: Participant-level columns that must be present.
        analysis_name: Name used in exclusion messages.

    Returns:
        The complete-case subset and a list of notes describing exclusions.
        Exclusions are also raised as warnings.
    """
    required_columns = [outcome, *covariates]
    exclusion_reasons = {}
    for participant_id, rows in analysis_data.groupby(PARTICIPANT):
        if set(rows[ARM]) != {OPERATED, NON_OPERATED}:
            exclusion_reasons[participant_id] = "does not have both arms"
            continue
        missing = [col for col in required_columns if rows[col].isna().any()]
        if missing:
            exclusion_reasons[participant_id] = f"missing {', '.join(missing)}"

    subset = analysis_data[~analysis_data[PARTICIPANT].isin(exclusion_reasons)]

    notes = []
    for participant_id, reason in exclusion_reasons.items():
        message = (
            f"{analysis_name}: participant {participant_id} excluded "
            f"({reason})."
        )
        warnings.warn(message)
        notes.append(message)
    return subset.copy(), notes


def _to_participant_level(
    complete_cases: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """Reshape complete-case long data to one row per participant.

    Pairing uses the participant identifier (`DataFrame.pivot`), never row
    order, so the operated and non-operated columns always belong to the
    same person.

    Args:
        complete_cases: Output of `_select_complete_cases`.
        outcome: Outcome column to reshape.

    Returns:
        DataFrame indexed by participant with columns `operated`,
        `non-operated`, `difference` (operated minus non-operated),
        `sex` and `rtsa_on_dominant_arm`.
    """
    paired = complete_cases.pivot(
        index=PARTICIPANT, columns=ARM, values=outcome
    )
    covariates = complete_cases.groupby(PARTICIPANT)[
        [SEX_CODE, DOMINANT_ARM_SURGERY]
    ].first()

    paired = paired[[OPERATED, NON_OPERATED]].join(covariates)
    paired[DIFFERENCE] = paired[OPERATED] - paired[NON_OPERATED]
    return paired


# -------------------------------------------------------------------
# Helper functions: shared result formatting
# -------------------------------------------------------------------

def _describe_groups(
    groups: dict[str, pd.Series],
    label_column: str,
) -> pd.DataFrame:
    """Return n, mean, standard deviation and coefficient of variation.

    Args:
        groups: Mapping of group label to the values in that group.
        label_column: Name of the column holding the group labels.

    Returns:
        DataFrame with one row per group.
    """
    return pd.DataFrame([
        {
            label_column: label,
            "n": values.shape[0],
            "Mean": values.mean(),
            "Std": values.std(),
            "CofV": values.std() / values.mean(),
        }
        for label, values in groups.items()
    ])


def _format_notes(notes: list[str]) -> str:
    """Join notes into one cell, or return a placeholder if there are none."""
    return " | ".join(notes) if notes else NO_NOTES


def _shapiro_p_value(values: pd.Series) -> float:
    """Return the Shapiro-Wilk p-value, or NaN if fewer than 3 values.

    This is reported as a diagnostic only. It is never used to choose a
    test, because with small samples it has little power and with large
    samples it flags trivial departures.
    """
    if len(values) < 3:
        return np.nan
    return stats.shapiro(values).pvalue


def _insufficient_data_note(n_participants: int) -> str | None:
    """Return a note if there are too few participants to run a test."""
    if n_participants < MIN_PARTICIPANTS_FOR_TEST:
        return (
            f"Not tested: only {n_participants} complete participant(s); "
            f"at least {MIN_PARTICIPANTS_FOR_TEST} are required."
        )
    return None


# -------------------------------------------------------------------
# Helper functions: t-tests
# -------------------------------------------------------------------

def _create_paired_ttest_table(
    analysis_data: pd.DataFrame,
    outcome: str,
) -> pd.DataFrame:
    """Compare operated and non-operated arms with a paired t-test.

    The paired t-test analyses each participant's operated minus
    non-operated difference, so between-participant variation cancels out.
    It assumes the differences are approximately normal; with n around 20
    it is reasonably robust to moderate skew. A Wilcoxon signed-rank test is
    reported alongside as a sensitivity check (it tests a shift in the
    distribution of differences rather than the mean difference).

    Args:
        analysis_data: Validated long-format analysis data.
        outcome: Outcome column to compare, e.g. ``rotation_rate``.

    Returns:
        Two rows (operated, non-operated) of descriptive statistics, followed
        by test results repeated on both rows: t-statistic, p-value, df, the
        mean paired difference with its confidence interval, diagnostics and
        notes. The first seven columns match the original table layout.
    """
    complete_cases, notes = _select_complete_cases(
        analysis_data, outcome, [], f"Paired t-test ({outcome})"
    )
    paired = _to_participant_level(complete_cases, outcome)

    table = _describe_groups(
        {"Operated": paired[OPERATED], "Non-operated": paired[NON_OPERATED]},
        label_column="Arm",
    )

    insufficient = _insufficient_data_note(len(paired))
    if insufficient:
        return table.assign(**{NOTES: _format_notes(notes + [insufficient])})

    test = stats.ttest_rel(paired[OPERATED], paired[NON_OPERATED])
    confidence_interval = test.confidence_interval(confidence_level=1 - ALPHA)
    differences = paired[DIFFERENCE]

    return table.assign(**{
        "t-statistic": test.statistic,
        "p-value": test.pvalue,
        "df": test.df,
        "Mean difference": differences.mean(),
        "Difference definition": "Operated - Non-operated (paired)",
        CI_LOWER: confidence_interval.low,
        CI_UPPER: confidence_interval.high,
        "Test": "Paired-samples t-test (scipy.stats.ttest_rel)",
        "Shapiro-Wilk p (paired differences)": _shapiro_p_value(differences),
        "Skewness (paired differences)": stats.skew(differences),
        "Wilcoxon signed-rank p (sensitivity)": stats.wilcoxon(
            differences
        ).pvalue,  # type: ignore --> linter throwing error because it sees a tuple, not a WilcoxonResult object, but this is the correct attribute
        NOTES: _format_notes(notes),
    })


# -------------------------------------------------------------------
# Helper functions: descriptive statistics (primary objective)
# -------------------------------------------------------------------

def _describe_distribution(values: pd.Series) -> dict[str, float]:
    """Summarise the distribution of one set of values.

    The mean's confidence interval is a t-interval
    (``scipy.stats.ttest_1samp``). The median, quartiles and range are
    reported alongside because rotation rates may be skewed.

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
        "Median": median,
        "Q1 (25th percentile)": q1,
        "Q3 (75th percentile)": q3,
        "Min": values.min(),
        "Max": values.max(),
        "Skewness": stats.skew(values) if n >= 3 else np.nan,
        "Shapiro-Wilk p": _shapiro_p_value(values),
    }


def _create_distribution_table(analysis_data: pd.DataFrame) -> pd.DataFrame:
    """Describe rotation rate overall and within each subgroup.

    The operated arm is the primary measure. The non-operated arm and the
    paired difference are included for context. Each arm uses every
    participant with that arm measured; the difference uses participants
    with both arms measured. Participants missing a subgroup value are left
    out of that subgroup's rows only.

    Returns:
        One row per subgroup and measure (see ``DESCRIPTIVE_SUBGROUPS``).
    """
    complete_pairs, _ = _select_complete_cases(
        analysis_data, ROTATION_RATE, [], "Paired-difference descriptives"
    )
    paired = _to_participant_level(complete_pairs, ROTATION_RATE)
    arm_rows = {
        arm: analysis_data[analysis_data[ARM] == arm].dropna(
            subset=[ROTATION_RATE]
        )
        for arm in (OPERATED, NON_OPERATED)
    }
    measures = {
        "Operated arm rotation rate": (arm_rows[OPERATED], ROTATION_RATE),
        "Non-operated arm rotation rate": (
            arm_rows[NON_OPERATED], ROTATION_RATE
        ),
        "Difference (operated - non-operated)": (paired, DIFFERENCE),
    }

    rows = []
    for subgroup_label, column, level in DESCRIPTIVE_SUBGROUPS:
        for measure_label, (table, value_column) in measures.items():
            in_subgroup = (
                table if column is None else table[table[column] == level]
            )
            rows.append({
                "Subgroup": subgroup_label,
                "Measure": measure_label,
                **_describe_distribution(in_subgroup[value_column]),
            })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# Helper functions: paired-difference regression models
# -------------------------------------------------------------------

def _effect_code_covariates(
    paired: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """Add -0.5/+0.5 effect-coded columns for each covariate.

    With this coding the intercept is the difference averaged with equal
    weight across covariate groups, and each coefficient is the full
    difference between that covariate's two groups (see ``EFFECT_CODES``).
    """
    coded = paired.copy()
    for covariate in covariates:
        coded_column, codes = EFFECT_CODES[covariate]
        coded[coded_column] = coded[covariate].map(codes)
    return coded


def _model_design_notes(
    paired: pd.DataFrame,
    covariates: list[str],
) -> tuple[list[str], list[str]]:
    """Check whether a model is identifiable and note sparse covariates.

    Args:
        paired: Participant-level data (one row per participant).
        covariates: Covariates in the model.

    Returns:
        A tuple ``(problems, notes)``. ``problems`` lists reasons the model
        cannot be fitted (empty if it can). ``notes`` lists warnings that do
        not prevent fitting.
    """
    n_participants = len(paired)
    n_parameters = 1 + len(covariates)  # intercept + covariates

    problems = []
    if n_participants <= n_parameters + 1:
        problems.append(
            f"Not fitted: {n_participants} participants is too few for "
            f"{n_parameters} parameters."
        )

    notes = []
    for covariate in covariates:
        level_counts = paired[covariate].value_counts()
        if len(level_counts) < 2:
            problems.append(
                f"Not fitted: '{covariate}' has only one observed level."
            )
            continue
        sparse_levels = level_counts[
            level_counts < MIN_PARTICIPANTS_PER_COVARIATE_LEVEL
        ]
        for level, count in sparse_levels.items():
            notes.append(
                f"Only {count} participants have {covariate} = {level}; "
                "estimates involving this covariate rest heavily on them."
            )
    return problems, notes


def _unfitted_model_row(
    model_details: dict[str, object],
    notes: list[str],
) -> pd.DataFrame:
    """Return a one-row table recording why a model was not fitted.

    Keeping this row in the output ensures a model is never silently
    omitted.
    """
    return pd.DataFrame([{**model_details, NOTES: _format_notes(notes)}])


def _fit_difference_model(
    analysis_data: pd.DataFrame,
    model_name: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Regress the paired rotation-rate difference on covariates.

    The model is `difference ~ covariates` (ordinary least squares,
    `statsmodels`), where difference is operated minus non-operated
    rotation rate for each participant. Each model uses its own
    complete-case subset. Inference uses t-tests with n - p degrees of
    freedom. p-values with HC3 heteroscedasticity-robust standard errors
    are reported as a sensitivity check, because the spread of the
    differences may not be equal across covariate groups.

    Args:
        analysis_data: Validated long-format analysis data.
        model_name: Label for the model, e.g. "B: sex".
        covariates: Participant-level covariates (raw column names).

    Returns:
        One row per term with the estimate, standard error, confidence
        interval, t-statistic, df, p-value, robust p-value, residual
        normality and notes. If the model cannot be fitted, a single row
        explains why.
    """
    complete_cases, notes = _select_complete_cases(
        analysis_data, ROTATION_RATE, covariates, f"Model {model_name}"
    )
    paired = _to_participant_level(complete_cases, ROTATION_RATE)
    coded_columns = [EFFECT_CODES[covariate][0] for covariate in covariates]
    formula = f"{DIFFERENCE} ~ " + (" + ".join(coded_columns) or "1")
    model_details = {
        "Model": model_name,
        "Formula": formula,
        "n participants": len(paired),
    }

    problems, design_notes = _model_design_notes(paired, covariates)
    notes += design_notes
    if problems:
        return _unfitted_model_row(model_details, notes + problems)

    model_data = _effect_code_covariates(paired, covariates)
    model = smf.ols(formula, model_data)
    result = model.fit()
    robust_result = model.fit(cov_type="HC3")

    terms = result.params.index
    confidence_intervals = result.conf_int(alpha=ALPHA)

    return pd.DataFrame({
        **model_details,
        "Term": [TERM_LABELS.get(term, term) for term in terms],
        "Estimate": result.params.values,
        "Std. error": result.bse.values,
        CI_LOWER: confidence_intervals[0].values,
        CI_UPPER: confidence_intervals[1].values,
        "t-statistic": result.tvalues.values,
        "df": result.df_resid,
        "p-value": result.pvalues.values,
        "HC3 robust p (sensitivity)": robust_result.pvalues[terms].values,
        "Residual Shapiro-Wilk p": _shapiro_p_value(result.resid),
        NOTES: _format_notes(notes),
    })


def _fit_difference_models(analysis_data: pd.DataFrame) -> pd.DataFrame:
    """Fit Models A-D (see ``DIFFERENCE_MODELS``) and stack their results."""
    return pd.concat(
        [
            _fit_difference_model(analysis_data, model_name, covariates)
            for model_name, covariates in DIFFERENCE_MODELS.items()
        ],
        ignore_index=True,
    )


# -------------------------------------------------------------------
# Helper functions: output
# -------------------------------------------------------------------

def _save_tables_to_excel(
    tables: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """Write each table to its own sheet, in the order given."""
    with pd.ExcelWriter(out_path) as writer:
        for sheet_name, table in tables.items():
            table.to_excel(writer, sheet_name=sheet_name, index=False)


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

    # 2. Long-format data for analysis (one row per arm), validated
    analysis_data = _validate_analysis_data(_create_analysis_dataframe(data))

    # 3. Primary objective: distribution of rotation rate
    rotation_rate_distribution = _create_distribution_table(analysis_data)

    # 4. Operated vs non-operated arms: paired t-tests
    cumulative_rotation_ttest = _create_paired_ttest_table(
        analysis_data, CUMULATIVE_ROTATION
    )
    rotation_rate_ttest = _create_paired_ttest_table(
        analysis_data, ROTATION_RATE
    )

    # 5. Operated vs non-operated difference, controlling for sex and/or
    #    dominant-arm surgery: paired-difference regressions A-D
    difference_model_results = _fit_difference_models(analysis_data)

    # 6. Save the tables to an Excel file
    _save_tables_to_excel(
        {
            "demographics": demographics_data,
            "rotation_data_by_arm": rotation_data,
            "rotation_rate_distribution": rotation_rate_distribution,
            "cumulative_rotation_stats": cumulative_rotation_ttest,
            "rotation_rate_stats": rotation_rate_ttest,
            "difference_model_stats": difference_model_results,
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
