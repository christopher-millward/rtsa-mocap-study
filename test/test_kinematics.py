from unittest.mock import call, patch

import numpy as np
import pytest

from utils.kinematics import (
    calculate_arm_rotations,
    calculate_rotation_angles,
    calculate_total_rotation,
    create_rotation_matrices,
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


# --- Building Matrices ---
# Extracting matrices should preserve the arm-specific column ordering frame by frame.
@pytest.mark.parametrize(
    ("arm",),
    [
        pytest.param("L", id="left"),
        pytest.param("R", id="right"),
    ],
)
def test_create_rotation_matrices_returns_expected_3d_array(arm):
    left_values = [
        [0.0, 0.1, 0.2],
        [0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8],
    ]
    right_values = [
        [1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5],
        [1.6, 1.7, 1.8],
    ]
    data = np.vstack(
        [
            _row_with_both_arms(left_values, right_values),
            _row_with_both_arms(right_values, left_values),
        ]
    )
    matrices = create_rotation_matrices(data, arm)

    expected = np.stack(
        [
            np.array(left_values if arm == "L" else right_values, dtype=np.float64),
            np.array(right_values if arm == "L" else left_values, dtype=np.float64),
        ]
    )

    assert matrices.shape == (2, 3, 3)
    assert np.array_equal(matrices, expected)

# Send invalid arm identifiers to the extractor and expect a ValueError.
@pytest.mark.parametrize("arm", [pytest.param("X", id="invalid-arm")])
def test_extract_rotation_matrix_rejects_invalid_arm(arm):
    with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
        create_rotation_matrices(np.zeros(18, dtype=np.float64), arm)


# --- Calculating Angles ---
# The angle calculation should follow the trace formula for a batch of matrices.
@pytest.mark.parametrize(
    ("rotation_matrices", "expected"),
    [
        pytest.param(np.stack([np.eye(3)]), np.array([0.0], dtype=np.float64), id="identity"),
        pytest.param(
            np.stack(
                [
                    np.array(
                        [
                            [1 / 3, 1 / 3, 1 / 3],
                            [1 / 3, 1 / 3, 1 / 3],
                            [1 / 3, 1 / 3, 1 / 3],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            np.array([np.pi / 2], dtype=np.float64),
            id="trace-one",
        ),
    ],
)
def test_calculate_rotation_angles(rotation_matrices, expected):
    angles = calculate_rotation_angles(rotation_matrices)

    assert np.allclose(angles, expected)


# --- Total Rotation ---
# Total rotation sums the angle for every frame in the input array.
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
