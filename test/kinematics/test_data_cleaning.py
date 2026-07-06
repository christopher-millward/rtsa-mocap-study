import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from utils.data_cleaning import align_axes_with_ISB


# ---- Global test variables ----
TOLERANCE = 1e-9
TORSO_TO_ISB = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
RIGHT_TO_ISB = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
LEFT_TO_ISB = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _expected_alignment(rotation_matrices, arm):
    arm_transform = LEFT_TO_ISB if arm == "left" else RIGHT_TO_ISB
    return TORSO_TO_ISB @ rotation_matrices @ arm_transform.T


class TestAlignAxesWithISB:

    # Should reject invalid arm identifiers.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("X", id="string"),
            pytest.param(2, id="integer"),
            pytest.param(None, id="none"),
        ],
    )
    def test_should_reject_invalid_arm(self, arm):
        matrices = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        with pytest.raises(ValueError, match="arm must be 'left' or 'right'"):
            align_axes_with_ISB(matrices, arm)

    # Should reject non-3x3 inputs.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros(3, dtype=np.float64), id="1d-array"),
            pytest.param(np.zeros((3, 3), dtype=np.float64), id="2d-array"),
            pytest.param(np.zeros((1, 1, 3, 3), dtype=np.float64), id="4d-array"),
            pytest.param(np.zeros((2, 3, 4), dtype=np.float64), id="non-square-matrices"),
        ],
    )
    def test_should_reject_invalid_shapes(self, rotation_matrices):
        with pytest.raises(ValueError, match=r"rotation_matrices must have shape \(n_frames, 3, 3\)"):
            align_axes_with_ISB(rotation_matrices, "left")

    # Should reject empty batches.
    def test_should_reject_empty_batch(self):
        rotation_matrices = np.zeros((0, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="rotation_matrices must contain at least one matrix"):
            align_axes_with_ISB(rotation_matrices, "left")

    # Should reject non-numeric values during float conversion.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.array([[[1, 0, 0], [0, None, 0], [0, 0, 1]]], dtype=object), id="none"),
            pytest.param(np.array([[[1, 0, 0], [0, "x", 0], [0, 0, 1]]], dtype=object), id="string"),
        ],
    )
    def test_should_reject_non_numeric_values(self, rotation_matrices):
        with pytest.raises(ValueError, match=r"rotation_matrices must contain only (numeric values|finite values)"):
            align_axes_with_ISB(rotation_matrices, "left")

    # Should reject non-finite numeric values.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.array([[[1, 0, 0], [0, np.nan, 0], [0, 0, 1]]], dtype=np.float64), id="nan"),
            pytest.param(np.array([[[1, 0, 0], [0, np.inf, 0], [0, 0, 1]]], dtype=np.float64), id="inf"),
        ],
    )
    def test_should_reject_non_finite_values(self, rotation_matrices):
        with pytest.raises(ValueError, match="rotation_matrices must contain only finite values"):
            align_axes_with_ISB(rotation_matrices, "left")

    # Should align the identity matrix correctly for both arms.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_align_identity_rotation(self, arm):
        rotation_matrices = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        expected = _expected_alignment(rotation_matrices, arm)

        result = align_axes_with_ISB(rotation_matrices, arm)

        assert np.allclose(result, expected)

    # Should align 90-degree rotations about all axes.
    @pytest.mark.parametrize(
        ("axis", "arm"),
        [
            pytest.param("X", "left", id="x-left"),
            pytest.param("Y", "left", id="y-left"),
            pytest.param("Z", "left", id="z-left"),
            pytest.param("X", "right", id="x-right"),
            pytest.param("Y", "right", id="y-right"),
            pytest.param("Z", "right", id="z-right"),
        ],
    )
    def test_should_align_90_degree_rotations(self, axis, arm):
        rotation_matrices = R.from_euler(axis, np.pi / 2).as_matrix().reshape(1, 3, 3)
        expected = _expected_alignment(rotation_matrices, arm)

        result = align_axes_with_ISB(rotation_matrices, arm)

        assert np.allclose(result, expected)

    # Should align batches of matrices frame by frame.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_align_multiple_matrices(self, arm):
        rotation_matrices = np.stack(
            [
                R.from_euler("X", 0.1).as_matrix(),
                R.from_euler("Y", 0.2).as_matrix(),
                R.from_euler("Z", 0.3).as_matrix(),
                R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix(),
            ],
            axis=0,
        )
        expected = _expected_alignment(rotation_matrices, arm)

        result = align_axes_with_ISB(rotation_matrices, arm)

        assert np.allclose(result, expected)

    # Should preserve orthonormality for valid rotation matrices.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_preserve_orthonormality(self, arm):
        rotation_matrices = R.random(8).as_matrix()
        result = align_axes_with_ISB(rotation_matrices, arm)

        gram = np.matmul(np.transpose(result, (0, 2, 1)), result)
        identity = np.broadcast_to(np.eye(3, dtype=np.float64), gram.shape)

        assert np.allclose(gram, identity, atol=TOLERANCE)

    # Should preserve a determinant of +1 for proper rotation matrices.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_preserve_determinant(self, arm):
        rotation_matrices = R.random(8).as_matrix()
        result = align_axes_with_ISB(rotation_matrices, arm)

        assert np.allclose(np.abs(np.linalg.det(result)), 1.0, atol=TOLERANCE)

    # Should recover the original matrices when the inverse transform is applied.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_recover_original_matrix_with_inverse_transform(self, arm):
        rotation_matrices = np.stack(
            [
                R.from_euler("X", 0.12).as_matrix(),
                R.from_euler("Y", 0.22).as_matrix(),
                R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix(),
            ],
            axis=0,
        )
        result = align_axes_with_ISB(rotation_matrices, arm)
        arm_transform = LEFT_TO_ISB if arm == "left" else RIGHT_TO_ISB
        recovered = TORSO_TO_ISB.T @ result @ arm_transform

        assert np.allclose(recovered, rotation_matrices, atol=TOLERANCE)

    # Should remain stable for rotations very close to identity.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_handle_rotations_close_to_identity(self, arm):
        rotation_matrices = R.from_euler("Y", 1e-10).as_matrix().reshape(1, 3, 3)
        expected = _expected_alignment(rotation_matrices, arm)

        result = align_axes_with_ISB(rotation_matrices, arm)

        assert np.allclose(result, expected, atol=TOLERANCE)

    # Should produce a different result when the transformation is applied twice.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_not_match_single_application_after_repeated_application(self, arm):
        rotation_matrices = R.from_euler("XYZ", [0.25, 0.5, 0.75]).as_matrix().reshape(1, 3, 3)
        single_application = align_axes_with_ISB(rotation_matrices, arm)
        repeated_application = align_axes_with_ISB(single_application, arm)

        assert not np.allclose(repeated_application, single_application)
