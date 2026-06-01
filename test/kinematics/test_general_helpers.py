import numpy as np
import pytest
from utils.kinematics.general_helpers import (
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

def _is_rotation_matrix(R):
    """Sanity check orthonormality."""
    return np.allclose(R.T @ R, np.eye(3), atol=1e-8)


class TestCreateRotationMatrices:
    # Extracting matrices should preserve the arm-specific column ordering frame by frame.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("L", id="left"),
            pytest.param("R", id="right"),
        ]
    )
    def test_should_return_expected_3d_array(self, arm):
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

    # Test that it actually builds valid rotation matrices, not just any 3x3 arrays.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("L", id="left"),
            pytest.param("R", id="right"),
        ],
    )
    def test_should_return_valid_rotation_matrices(self,arm):
        """Each extracted frame should still be a valid rotation matrix."""
        left_frames = [
            np.eye(3, dtype=np.float64),
            _rotation_matrix_x(np.pi / 6),
        ]
        right_frames = [
            _rotation_matrix_y(np.pi / 4),
            _rotation_matrix_z(np.pi / 3),
        ]
        data = np.vstack(
            [
                _row_with_both_arms(left_frames[0], right_frames[0]),
                _row_with_both_arms(left_frames[1], right_frames[1]),
            ]
        )

        matrices = create_rotation_matrices(data, arm)

        assert matrices.shape == (2, 3, 3)
        assert all(_is_rotation_matrix(matrix) for matrix in matrices)

    # Send invalid arm identifiers to the extractor and expect a ValueError.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("X", id="string"),
            pytest.param(2, id="dtype"),
            pytest.param(None, id="NoneType")
        ]
    )
    def test_should_reject_invalid_arm(self, arm):
        with pytest.raises(ValueError, match="arm must be 'L' or 'R'"):
            create_rotation_matrices(np.zeros(18, dtype=np.float64), arm)

