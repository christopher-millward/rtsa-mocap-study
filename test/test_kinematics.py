from unittest.mock import call, patch

import numpy as np
import pytest

from utils.kinematics import (
    calculate_arm_rotations,
    calculate_rotation_angle,
    calculate_total_rotation,
    extract_rotation_matrix,
)


# ---- Helper Functions ----
def _matrix_row(values):
    """Build a 1D NumPy row with the nine matrix entries for one arm."""
    return np.array([values[i][j] for i in range(3) for j in range(3)], dtype=np.float64)


def _row_with_both_arms(left_values, right_values):
    """Build a 1D row with left-arm values followed by right-arm values."""
    return np.concatenate((_matrix_row(left_values), _matrix_row(right_values)))


def _rotation_matrix_z(angle):
    """Build a right-handed Z-axis rotation matrix for a given angle."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


# --- Building Matrix ---
# Extracting a matrix should preserve the arm-specific column ordering.
@pytest.mark.parametrize(
    ("arm",),
    [
        pytest.param("L", id="left"),
        pytest.param("R", id="right"),
    ],
)
def test_extract_rotation_matrix_returns_expected_3x3_matrix(arm):
    values = [
        [0.0, 0.1, 0.2],
        [0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8],
    ]
    row = _row_with_both_arms(values, values)
    matrix = extract_rotation_matrix(row, arm)

    assert matrix.shape == (3, 3)
    assert np.array_equal(matrix, np.array(values, dtype=np.float64))

# Send invalid arm identifiers to the extractor and expect a ValueError.
@pytest.mark.parametrize("arm", [pytest.param("X", id="invalid-arm")])
def test_extract_rotation_matrix_rejects_invalid_arm(arm):
    with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
        extract_rotation_matrix(np.zeros(18, dtype=np.float64), arm)


# --- Calculating Angles ---
# The angle calculation should follow the trace formula.
@pytest.mark.parametrize(
    ("rotation_matrix", "expected"),
    [
        pytest.param(np.eye(3), 0.0, id="identity"),
        pytest.param(
            np.array(
                [
                    [1 / 3, 1 / 3, 1 / 3],
                    [1 / 3, 1 / 3, 1 / 3],
                    [1 / 3, 1 / 3, 1 / 3],
                ],
                dtype=np.float64,
            ),
            np.pi / 2,
            id="trace-one",
        ),
    ],
)
def test_calculate_rotation_angle(rotation_matrix, expected):
    angle = calculate_rotation_angle(rotation_matrix)

    assert angle == pytest.approx(expected)


# --- Total Rotation ---
# Total rotation sums the angle for every row in the input frame.
@pytest.mark.parametrize(
    ("arm", "n_samples", "target_rotation_magnitude"),
    [
        pytest.param("L", 10, np.pi / 2, id="left"),
        pytest.param("R", 10, np.pi / 2, id="right"),
        pytest.param("L", 10, np.pi * 3, id="left"),
        pytest.param("R", 10, 0, id="right"),
    ],
)
def test_calculate_total_rotation_sums_all_frames(arm, n_samples, target_rotation_magnitude):
    target_rotation_per_sample = target_rotation_magnitude / n_samples
    target_rotation_matrix = _rotation_matrix_z(target_rotation_per_sample)

    rows = np.vstack(
        [
            _row_with_both_arms(target_rotation_matrix, target_rotation_matrix)
            for _ in range(n_samples)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)

    assert total_rotation == pytest.approx(target_rotation_magnitude)

@pytest.mark.parametrize("arm", [pytest.param("Z", id="invalid-arm")])
def test_calculate_total_rotation_rejects_invalid_arm(arm):
    with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
        calculate_total_rotation(np.zeros((0, 18), dtype=np.float64), arm)


# --- Get Rotations for Each Arm ---
# The file-level helper should load once and delegate to both arm totals.
def test_calculate_arm_rotations_returns_both_arms():
    frame = np.vstack([_row_with_both_arms(np.eye(3), np.eye(3))])

    # calculate_arm_rotations now accepts the NumPy array loaded from np.loadtxt.
    with patch("utils.kinematics.calculate_total_rotation", side_effect=[1.25, 2.5]) as mock_total_rotation:
        left_rotation, right_rotation = calculate_arm_rotations(frame)

    assert mock_total_rotation.call_args_list == [call(frame, "L"), call(frame, "R")]
    assert left_rotation == 1.25
    assert right_rotation == 2.5


def test_calculate_arm_rotations_wraps_exceptions_from_total_rotation():
    # If calculate_total_rotation raises, calculate_arm_rotations should
    # wrap it as a ValueError with a helpful message.
    with patch("utils.kinematics.calculate_total_rotation", side_effect=Exception("boom")):
        with pytest.raises(ValueError, match="Failed to parse motion capture data"):
            calculate_arm_rotations(np.zeros((0, 18), dtype=np.float64))
