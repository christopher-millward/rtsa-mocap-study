"""
Functions to perform all statistics and output into an excel file.

Author: Christopher Millward

Research questions
------------------
1. Do rotation rates differ between operated and non-operated arms?
2. Do rotation rates differ between men and women?
3. Does the operated vs non-operated difference hold after controlling for sex?
4. Does the operated vs non-operated difference hold after controlling for whether the RTSA was on the dominant arm?
5. Does the operated vs non-operated difference hold after controlling for both sex and dominant-arm surgery?

Data structure
--------------
Every participant contributes one measurement from each arm, so the two
measurements from one participant are not independent. Sex and "RTSA on
dominant arm" are participant-level characteristics: they are constant
across a participant's two arms.

Statistical approach
--------------------
* Operated vs non-operated (Q1): paired-samples t-test
  (``scipy.stats.ttest_rel``). Pairs are matched on participant identity.
* Men vs women (Q2): Welch's independent-samples t-test
  (``scipy.stats.ttest_ind(equal_var=False)``) on each participant's mean
  rotation rate across both arms. Averaging the arms first gives exactly one
  independent value per participant, so the two arms of one person are not
  counted as two independent people. Welch's test does not assume equal
  group sizes or equal variances.
* Adjusted comparisons (Q3-Q5): linear mixed-effects models
  (``statsmodels`` ``MixedLM``, REML) with a random intercept per
  participant. A mixed model is preferred over repeated-measures ANOVA
  because it accepts participant-level covariates directly, reports the
  operated-arm effect as an interpretable estimate with a confidence
  interval, and does not discard a participant's data if a covariate is
  missing for a different model.

      Model A: rotation_rate ~ arm
      Model B: rotation_rate ~ arm + sex
      Model C: rotation_rate ~ arm + rtsa_on_dominant_arm
      Model D: rotation_rate ~ arm + sex + rtsa_on_dominant_arm

  Important: because sex and dominant-arm surgery do not vary within a
  participant, adding them to an additive model cannot change the estimated
  operated-arm effect when every participant has both arms measured. They
  only explain differences *between* participants. Models B-D therefore
  confirm, rather than alter, the Model A estimate. Asking whether the
  operated-arm effect *differs* by dominant-arm surgery would require an
  arm x dominant-arm-surgery interaction, which is deliberately not fitted
  here (see ``_describe_differences_by_subgroup`` for a descriptive view).

Multiple comparisons
--------------------
No p-value adjustment is applied. Q1 and Q2 are distinct, pre-specified
questions. Models A-D are sensitivity analyses of one parameter (the
operated-arm effect), not independent hypotheses, so correcting across them
would only inflate the p-value of a single question.

Age is intentionally not analyzed. All tests use alpha = 0.05.
"""
from __future__ import annotations

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
VALID_SEX_CODES = ("F", "M")  # The first code is the reference level.

# Columns added when the long data are reshaped to one row per participant.
DIFFERENCE = "difference"
PARTICIPANT_MEAN = "participant_mean"

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

# Covariates included in each mixed-effects model, in addition to arm.
MIXED_MODELS: dict[str, list[str]] = {
    "A: arm": [],
    "B: arm + sex": [SEX_CODE],
    "C: arm + dominant-arm surgery": [DOMINANT_ARM_SURGERY],
    "D: arm + sex + dominant-arm surgery": [SEX_CODE, DOMINANT_ARM_SURGERY],
}

# Human-readable labels for the coefficient names produced by the formulas.
TERM_LABELS = {
    "Intercept": "Intercept (non-operated arm, reference levels)",
    f"{ARM}[T.{OPERATED}]": "Arm: operated vs non-operated",
    f"{SEX_CODE}[T.M]": "Sex: M vs F",
    f"{DOMINANT_ARM_SURGERY}[T.True]": "RTSA on dominant arm: True vs False",
}


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


def _to_finite_float(value: object) -> float:
    """Convert a measurement to float, returning NaN if missing or infinite.

    Raises:
        TypeError or ValueError: If the value cannot be read as a number
            because that indicates a data problem rather than a missing
            measurement.
    """
    if value is None:
        return np.nan
    number = float(value) if isinstance(
        value, (int, float, np.number)) else np.nan
    return number if np.isfinite(number) else np.nan


def _create_analysis_dataframe(
    data: list[ParticipantDetails]
) -> pd.DataFrame:
    """Create a long-format DataFrame with one row per arm per participant.

    Participant identifiers match those used in the demographics and
    rotation-data tables (the participant's position in ``data``).

    Args:
        data: List of ParticipantDetails objects.

    Returns:
        DataFrame with the columns ``participant``, ``filename``, ``arm``
        ("operated" or "non-operated"), ``rotation_rate``,
        ``cumulative_rotation``, ``sex`` and ``rtsa_on_dominant_arm``.
        Missing or non-finite outcome values are stored as NaN. Sex and
        dominant-arm surgery are stored exactly as recorded; they are checked
        by ``_validate_analysis_data``.
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
                ROTATION_RATE: _to_finite_float(arm.humerothoracic.rotation_rate),
                CUMULATIVE_ROTATION: _to_finite_float(arm.humerothoracic.trace_total),
                SEX_CODE: participant.sex,
                DOMINANT_ARM_SURGERY: participant.rtsa_on_dominant_arm,
            })

    return pd.DataFrame(rows)


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
        analysis_data: Output of ``_create_analysis_dataframe``.

    Returns:
        A validated copy of ``analysis_data``.

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

    Pairing uses the participant identifier (``DataFrame.pivot``), never row
    order, so the operated and non-operated columns always belong to the
    same person.

    Args:
        complete_cases: Output of ``_select_complete_cases``.
        outcome: Outcome column to reshape.

    Returns:
        DataFrame indexed by participant with columns ``operated``,
        ``non-operated``, ``difference`` (operated minus non-operated),
        ``participant_mean`` (mean of both arms), ``sex`` and
        ``rtsa_on_dominant_arm``.
    """
    paired = complete_cases.pivot(
        index=PARTICIPANT, columns=ARM, values=outcome
    )
    covariates = complete_cases.groupby(PARTICIPANT)[
        [SEX_CODE, DOMINANT_ARM_SURGERY]
    ].first()

    paired = paired[[OPERATED, NON_OPERATED]].join(covariates)
    paired[DIFFERENCE] = paired[OPERATED] - paired[NON_OPERATED]
    paired[PARTICIPANT_MEAN] = paired[[OPERATED, NON_OPERATED]].mean(axis=1)
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
        ).pvalue,
        NOTES: _format_notes(notes),
    })


def _create_sex_comparison_table(analysis_data: pd.DataFrame) -> pd.DataFrame:
    """Compare men and women with Welch's t-test on participant means.

    Each participant has two arm measurements, which are correlated. Treating
    them as independent would double the apparent sample size and understate
    the uncertainty. Averaging the two arms first gives one independent
    value per participant. Because every participant has one operated and
    one non-operated arm, this mean is not biased by arm status. The
    estimate equals the sex coefficient of mixed Model B when all
    participants have both arms.

    Welch's test does not assume equal group sizes or equal variances.

    Args:
        analysis_data: Validated long-format analysis data.

    Returns:
        Two rows (F, M) of descriptive statistics of the participant-mean
        rotation rate, followed by test results repeated on both rows:
        t-statistic, p-value, Welch df, mean difference (M - F) with its
        confidence interval, and notes.
    """
    complete_cases, notes = _select_complete_cases(
        analysis_data, ROTATION_RATE, [SEX_CODE], "Sex comparison"
    )
    participant_means = _to_participant_level(complete_cases, ROTATION_RATE)
    means_by_sex = {
        sex_code: participant_means.loc[
            participant_means[SEX_CODE] == sex_code, PARTICIPANT_MEAN
        ]
        for sex_code in VALID_SEX_CODES
    }
    female_means, male_means = means_by_sex["F"], means_by_sex["M"]

    table = _describe_groups(means_by_sex, label_column=SEX)

    smallest_group = min(len(female_means), len(male_means))
    insufficient = _insufficient_data_note(smallest_group)
    if insufficient:
        return table.assign(**{NOTES: _format_notes(notes + [insufficient])})

    test = stats.ttest_ind(male_means, female_means, equal_var=False)
    confidence_interval = test.confidence_interval(confidence_level=1 - ALPHA)

    return table.assign(**{
        "t-statistic": test.statistic,
        "p-value": test.pvalue,
        "df": test.df,
        "Mean difference": male_means.mean() - female_means.mean(),
        "Difference definition": "M - F (participant mean of both arms)",
        CI_LOWER: confidence_interval.low,
        CI_UPPER: confidence_interval.high,
        "Test": "Welch's t-test (scipy.stats.ttest_ind, equal_var=False)",
        NOTES: _format_notes(notes),
    })


# -------------------------------------------------------------------
# Helper functions: linear mixed-effects models
# -------------------------------------------------------------------

def _prepare_model_frame(complete_cases: pd.DataFrame) -> pd.DataFrame:
    """Set explicit reference levels for the categorical model terms.

    Reference levels: non-operated arm, female, RTSA not on dominant arm.
    Coefficients are therefore operated minus non-operated, M minus F and
    dominant minus non-dominant surgery.
    """
    return complete_cases.assign(**{
        ARM: pd.Categorical(
            complete_cases[ARM], categories=[NON_OPERATED, OPERATED]
        ),
        SEX_CODE: pd.Categorical(
            complete_cases[SEX_CODE], categories=list(VALID_SEX_CODES)
        ),
        DOMINANT_ARM_SURGERY: pd.Categorical(
            complete_cases[DOMINANT_ARM_SURGERY], categories=[False, True]
        ),
    })


def _model_design_notes(
    complete_cases: pd.DataFrame,
    covariates: list[str],
) -> tuple[list[str], list[str]]:
    """Check whether a model is identifiable and note sparse covariates.

    Returns:
        A tuple ``(problems, notes)``. ``problems`` lists reasons the model
        cannot be fitted (empty if it can). ``notes`` lists warnings that do
        not prevent fitting.
    """
    participants = complete_cases.drop_duplicates(PARTICIPANT)
    n_participants = len(participants)
    n_fixed_effects = 2 + len(covariates)  # intercept + arm + covariates

    problems = []
    if n_participants <= n_fixed_effects:
        problems.append(
            f"Not fitted: {n_participants} participants is too few for "
            f"{n_fixed_effects} fixed effects."
        )

    notes = []
    for covariate in covariates:
        level_counts = participants[covariate].value_counts()
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
                "this covariate's own coefficient is imprecise."
            )

    if covariates:
        notes.append(
            "Covariates are participant-level, so with complete pairs they "
            "cannot change the arm estimate; they explain between-participant "
            "variation only."
        )
    return problems, notes


def _fit_mixed_model(
    analysis_data: pd.DataFrame,
    model_name: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Fit one random-intercept linear mixed model of rotation rate.

    The model is ``rotation_rate ~ arm + covariates + (1 | participant)``,
    fitted by REML with ``statsmodels``. Each model uses its own
    complete-case subset.

    p-values and confidence intervals are Wald z-based (normal
    approximation), which is what ``statsmodels`` provides; no
    small-sample degrees of freedom (Satterthwaite or Kenward-Roger) are
    available. With complete pairs, the paired t-test is the exact
    small-sample equivalent of Model A's arm effect.

    Args:
        analysis_data: Validated long-format analysis data.
        model_name: Label for the model, e.g. "A: arm".
        covariates: Participant-level covariates to add to arm.

    Returns:
        One row per fixed-effect term with the estimate, standard error,
        confidence interval, z-statistic and p-value, plus model-level
        diagnostics (sample size, variance components, convergence,
        residual and random-intercept normality) and notes. If the model
        cannot be fitted, a single row explains why.
    """
    formula = f"{ROTATION_RATE} ~ " + " + ".join([ARM, *covariates])
    complete_cases, notes = _select_complete_cases(
        analysis_data, ROTATION_RATE, covariates, f"Model {model_name}"
    )
    model_details = {
        "Model": model_name,
        "Formula": f"{formula} + (1 | {PARTICIPANT})",
        "n participants": complete_cases[PARTICIPANT].nunique(),
    }

    problems, design_notes = _model_design_notes(complete_cases, covariates)
    notes += design_notes
    if problems:
        return _unfitted_model_row(model_details, notes + problems)

    model_frame = _prepare_model_frame(complete_cases)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            result = smf.mixedlm(
                formula, model_frame, groups=model_frame[PARTICIPANT]
            ).fit(reml=True)
        except (np.linalg.LinAlgError, ValueError) as error:
            return _unfitted_model_row(
                model_details, notes + [f"Not fitted: {error}"]
            )
    notes += [f"statsmodels: {warning.message}" for warning in caught_warnings]

    random_intercept_variance = float(result.cov_re.iloc[0, 0])
    if random_intercept_variance <= 1e-8 * result.scale:
        notes.append(
            "Random-intercept variance is at the boundary (~0); the fit is "
            "singular and the model reduces to ordinary regression."
        )

    terms = result.fe_params.index
    confidence_intervals = result.conf_int(alpha=ALPHA).loc[terms]
    random_intercepts = pd.Series(
        [effects.iloc[0] for effects in result.random_effects.values()]
    )

    return pd.DataFrame({
        **model_details,
        "Term": [TERM_LABELS.get(term, term) for term in terms],
        "Estimate": result.fe_params.values,
        "Std. error": result.bse_fe.values,
        CI_LOWER: confidence_intervals[0].values,
        CI_UPPER: confidence_intervals[1].values,
        "z-statistic": result.tvalues[terms].values,
        "p-value": result.pvalues[terms].values,
        "p-value method": "Wald z-test (normal approximation), REML fit",
        "n observations": result.nobs,
        "Random intercept variance": random_intercept_variance,
        "Residual variance": result.scale,
        "Intraclass correlation": random_intercept_variance
        / (random_intercept_variance + result.scale),
        "Converged": result.converged,
        "Residual Shapiro-Wilk p": _shapiro_p_value(result.resid),
        "Random intercept Shapiro-Wilk p": _shapiro_p_value(random_intercepts),
        NOTES: _format_notes(notes),
    })


def _unfitted_model_row(
    model_details: dict[str, object],
    notes: list[str],
) -> pd.DataFrame:
    """Return a one-row table recording why a model was not fitted.

    Keeping this row in the output ensures a model is never silently
    omitted.
    """
    return pd.DataFrame([{**model_details, NOTES: _format_notes(notes)}])


def _fit_mixed_models(analysis_data: pd.DataFrame) -> pd.DataFrame:
    """Fit mixed Models A-D (see ``MIXED_MODELS``) and stack their results.

    Interactions are intentionally not included; all models are additive.
    """
    return pd.concat(
        [
            _fit_mixed_model(analysis_data, model_name, covariates)
            for model_name, covariates in MIXED_MODELS.items()
        ],
        ignore_index=True,
    )


# -------------------------------------------------------------------
# Helper functions: descriptive subgroup summary
# -------------------------------------------------------------------

def _describe_differences_by_subgroup(
    analysis_data: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise the operated minus non-operated difference by subgroup.

    Descriptive only; no hypothesis test is performed. The table shows
    whether the arm difference looks similar across sexes and across
    dominant- vs non-dominant-arm surgery, which the additive mixed models
    cannot reveal. Participants missing a subgroup value are left out of
    that subgroup's rows only.

    Returns:
        One row per subgroup level with the number of participants and the
        mean and standard deviation of the paired difference.
    """
    complete_cases, _ = _select_complete_cases(
        analysis_data, ROTATION_RATE, [], "Subgroup summary"
    )
    paired = _to_participant_level(complete_cases, ROTATION_RATE)

    summaries = []
    for subgroup, label in [
        (SEX_CODE, "Sex"),
        (DOMINANT_ARM_SURGERY, "RTSA on dominant arm"),
    ]:
        summary = (
            paired.groupby(subgroup)[DIFFERENCE]
            .agg(["count", "mean", "std"])
            .reset_index(names="Level")
            .assign(Subgroup=label)
        )
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True).rename(columns={
        "count": "n participants",
        "mean": "Mean difference (operated - non-operated)",
        "std": "Std of difference",
    })[[
        "Subgroup",
        "Level",
        "n participants",
        "Mean difference (operated - non-operated)",
        "Std of difference",
    ]]


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

    # 2. Long-format data for inference (one row per arm), validated
    analysis_data = _validate_analysis_data(_create_analysis_dataframe(data))

    # 3. Operated vs non-operated arms: paired t-tests
    cumulative_rotation_ttest = _create_paired_ttest_table(
        analysis_data, CUMULATIVE_ROTATION
    )
    rotation_rate_ttest = _create_paired_ttest_table(
        analysis_data, ROTATION_RATE
    )

    # 4. Men vs women: Welch's t-test on participant means
    sex_comparison = _create_sex_comparison_table(analysis_data)

    # 5. Operated vs non-operated, adjusted for sex and/or dominant-arm
    #    surgery: linear mixed models A-D
    mixed_model_results = _fit_mixed_models(analysis_data)

    # 6. Descriptive look at the arm difference within subgroups
    subgroup_differences = _describe_differences_by_subgroup(analysis_data)

    # 7. Save the tables to an Excel file
    _save_tables_to_excel(
        {
            "demographics": demographics_data,
            "rotation_data_by_arm": rotation_data,
            "cumulative_rotation_stats": cumulative_rotation_ttest,
            "rotation_rate_stats": rotation_rate_ttest,
            "sex_comparison_stats": sex_comparison,
            "mixed_model_stats": mixed_model_results,
            "arm_difference_by_subgroup": subgroup_differences,
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
