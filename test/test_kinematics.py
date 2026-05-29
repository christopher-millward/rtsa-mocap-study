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

def _rotation_matrix_x(angle: float) -> np.ndarray:
    """Isolated rotation about the x-axis."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ],
        dtype=np.float64,
    )

def _rotation_matrix_y(angle: float) -> np.ndarray:
    """Isolated rotation about the y-axis."""
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )

def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Isolated rotation about the z-axis."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

def _rotation_matrix_xyz(
    x_angle: float,
    y_angle: float,
    z_angle: float,
) -> np.ndarray:
    """
    Intrinsic XYZ rotation composition.
    """

    rx = _rotation_matrix_x(x_angle)
    ry = _rotation_matrix_y(y_angle)
    rz = _rotation_matrix_z(z_angle)

    return rz @ ry @ rx


# --- Building Matrices ---
# Extracting matrices should preserve the arm-specific column ordering frame by frame.
@pytest.mark.parametrize(
    "arm",
    [
        pytest.param("L", id="left"),
        pytest.param("R", id="right"),
    ]
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
            np.array(left_values if arm ==
                     "L" else right_values, dtype=np.float64),
            np.array(right_values if arm ==
                     "L" else left_values, dtype=np.float64),
        ]
    )
    assert matrices.shape == (2, 3, 3)
    assert np.array_equal(matrices, expected)

# Send invalid arm identifiers to the extractor and expect a ValueError.
@pytest.mark.parametrize(
    "arm",
    [
        pytest.param("X", id="string"),
        pytest.param(2, id="dtype"),
        pytest.param(None, id="NoneType")
    ]
)
def test_extract_rotation_matrix_rejects_invalid_arm(arm):
    with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
        create_rotation_matrices(np.zeros(18, dtype=np.float64), arm)


# --- Calculating Angles ---
# The angle calculation should follow the trace formula for a batch of matrices.
@pytest.mark.parametrize(
    ("rotation_matrices", "expected"),
    [
        pytest.param(
            np.stack([np.eye(3)]),
            np.array([0.0], dtype=np.float64),
            id="identity"
        ),
        pytest.param(
            np.stack([np.full((3, 3), 1/3, dtype=np.float64)]),  # all vals 1/3
            np.array([np.pi / 2], dtype=np.float64),  # results in 90 deg rot
            id="trace-one"
        )
    ]
)
def test_calculate_rotation_angles(rotation_matrices, expected):
    angles = calculate_rotation_angles(rotation_matrices)
    assert np.allclose(angles, expected)


# --- Total Rotation ---
# Total rotation sums the angle for every frame in the input array.
@pytest.mark.parametrize(
    ("arm", "rotation_builder", "target_angle", "n_samples"),
    [
        pytest.param("L", _rotation_matrix_x, np.pi / 2, 10, id="left-x-90"),
        pytest.param("R", _rotation_matrix_x, np.pi, 5, id="right-x-180"),
        pytest.param("L", _rotation_matrix_y, np.pi / 2, 10, id="left-y-90"),
        pytest.param("R", _rotation_matrix_y, np.pi / 4, 8, id="right-y-45"),
        pytest.param("L", _rotation_matrix_z, np.pi / 2, 10, id="left-z-90"),
        pytest.param("R", _rotation_matrix_z, np.pi * 0.75, 12, id="right-z-135"),
    ],
)
def test_calculate_total_rotation_single_axis(
    arm,
    rotation_builder,
    target_angle,
    n_samples,
):
    per_frame_angle = target_angle / n_samples
    rotation_matrix = rotation_builder(per_frame_angle)
    rows = np.vstack(
        [
            _row_with_both_arms(
                rotation_matrix,
                rotation_matrix,
            )
            for _ in range(n_samples)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)
    assert total_rotation == pytest.approx(target_angle)

# Test that rotation about all axes will get summed.
@pytest.mark.parametrize(
    ("arm", "x_angle", "y_angle", "z_angle", "n_samples"),
    [
        pytest.param("L", 0.1, 0.2, 0.3, 10, id="left-small-xyz"),
        pytest.param("R", np.pi / 6, np.pi / 8, np.pi / 10, 20, id="right-medium-xyz"),
        pytest.param("L", np.pi / 3, np.pi / 4, np.pi / 5, 5, id="left-large-xyz"),
    ],
)
def test_calculate_total_rotation_xyz_composition(
    arm,
    x_angle,
    y_angle,
    z_angle,
    n_samples,
):
    per_frame_matrix = _rotation_matrix_xyz(
        x_angle,
        y_angle,
        z_angle,
    )

    trace = np.trace(per_frame_matrix)
    expected_per_frame_angle = np.arccos(
        np.clip((trace - 1) / 2, -1.0, 1.0)
    )

    rows = np.vstack(
        [
            _row_with_both_arms(
                per_frame_matrix,
                per_frame_matrix,
            )
            for _ in range(n_samples)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)
    expected_total = expected_per_frame_angle * n_samples
    assert total_rotation == pytest.approx(expected_total)

# Test that zero rotation is handled correctly.
@pytest.mark.parametrize("arm", ["L", "R"])
def test_calculate_total_rotation_zero_rotation(arm):
    identity = np.eye(3)
    rows = np.vstack(
        [
            _row_with_both_arms(identity, identity)
            for _ in range(10)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)
    assert total_rotation == pytest.approx(0.0)

# Test stability near zero rotation.
@pytest.mark.parametrize("arm", ["L", "R"])
def test_calculate_total_rotation_tiny_rotations(arm):
    tiny_angle = 1e-9
    rotation_matrix = _rotation_matrix_z(tiny_angle)
    rows = np.vstack(
        [
            _row_with_both_arms(
                rotation_matrix,
                rotation_matrix,
            )
            for _ in range(100)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)
    assert total_rotation >= 0.0

# Test stability near pi rotation.
@pytest.mark.parametrize("arm", ["L", "R"])
def test_calculate_total_rotation_near_pi(arm):
    angle = np.pi - 1e-8
    n_rows = 3
    rotation_matrix = _rotation_matrix_y(angle)
    rows = np.vstack(
        [
            _row_with_both_arms(
                rotation_matrix,
                rotation_matrix,
            )
            for _ in range(n_rows)
        ]
    )

    total_rotation = calculate_total_rotation(rows, arm)
    expected = angle * n_rows
    assert total_rotation == pytest.approx(expected)

# Test negative angles are still summed as positive rotation.
@pytest.mark.parametrize("arm", ["L", "R"])
def test_calculate_total_rotation_negative_angles_match_positive(arm):
    positive = _rotation_matrix_x(np.pi / 4)
    negative = _rotation_matrix_x(-np.pi / 4)
    n_rows = 5

    positive_rows = np.vstack(
        [
            _row_with_both_arms(positive, positive)
            for _ in range(n_rows)
        ]
    )

    negative_rows = np.vstack(
        [
            _row_with_both_arms(negative, negative)
            for _ in range(n_rows)
        ]
    )

    positive_total = calculate_total_rotation(positive_rows, arm)
    negative_total = calculate_total_rotation(negative_rows, arm)
    assert positive_total == pytest.approx(negative_total)

# Send invalid arm identifiers and expect a ValueError
@pytest.mark.parametrize(
    "arm",
    [
        pytest.param("X", id="string"),
        pytest.param(2, id="dtype"),
        pytest.param(None, id="NoneType")
    ]
)
def test_calculate_total_rotation_rejects_invalid_arm(arm):
    with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
        calculate_total_rotation(np.zeros((0, 18), dtype=np.float64), arm)


# --- Get Rotations for Each Arm ---
# The file-level helper should load once and delegate to both arm totals.
def test_calculate_arm_rotations_returns_both_arms():
    test_data = np.vstack([_row_with_both_arms(np.eye(3), np.eye(3))])

    # replace calculate_total_rotation with a mock that returns fixed values for left and right arms
    with patch("utils.kinematics.calculate_total_rotation", side_effect=[1.25, 2.5]) as mock_total_rotation:
        left_rotation, right_rotation = calculate_arm_rotations(test_data)

    # check that calculate_total_rotation was called with the correct arguments for each arm
    assert mock_total_rotation.call_args_list == [
        call(test_data, "L"), call(test_data, "R")]
    
    # check that the returned rotations match the mock values
    assert left_rotation == 1.25
    assert right_rotation == 2.5

def test_calculate_arm_rotations_wraps_exceptions_from_total_rotation():
    # If calculate_total_rotation raises, calculate_arm_rotations should
    # wrap it as a ValueError with a helpful message.
    with patch("utils.kinematics.calculate_total_rotation", side_effect=Exception("Boom!")):
        with pytest.raises(ValueError, match="Failed to parse motion capture data"):
            calculate_arm_rotations(np.zeros((0, 18), dtype=np.float64))
