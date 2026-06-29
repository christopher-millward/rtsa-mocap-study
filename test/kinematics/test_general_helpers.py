import numpy as np
import pytest
from unittest.mock import patch
from scipy.spatial.transform import Rotation as R
from utils.kinematics.general_helpers import (
    validate_orthonorm_and_det,
    create_rotation_matrices,
)


# ---- Global test variables ----
small_angle = 1e-3
tolerance = 1e-9
singularity_tolerance = 1e-7


# ---- Tests ----
class TestValidateOrthonormAndDet:
    # Should reject non-3x3 matrices
    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64), id="2x2-matrices"),
            pytest.param(np.zeros((2, 3, 4), dtype=np.float64), id="non-square-matrices"),
            pytest.param(np.zeros((2, 3), dtype=np.float64), id="flat-matrices"),
            pytest.param(np.zeros((2, 4, 4), dtype=np.float64), id="4x4-matrices"),
        ],
    )
    def test_should_reject_non_3x3_matrices(self, matrices):
        with pytest.raises(ValueError, match=r"matrices must have shape \(n_frames, 3, 3\)"):
            validate_orthonorm_and_det(matrices)

    # Should reject empty batch
    def test_should_reject_empty_batch(self):
        matrices = np.zeros((0, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="batch must contain at least one matrix"):
            validate_orthonorm_and_det(matrices)

    # Should coerce to float64
    def test_should_perform_calculations_using_float64_precision(self):
        data = R.from_euler("X", 0.1).as_matrix().reshape(1, 3, 3)

        # spy on the np.asarray call
        with patch("numpy.asarray", wraps=np.asarray) as spy_asarray:
            validate_orthonorm_and_det(data)

        # assert that it is specifically being called on the data
        # and not in some other internal call
        assert any(
            call.args[0] is data and call.kwargs.get("dtype") == np.float64
            for call in spy_asarray.call_args_list
    )

    # Should accept valid rotation matrix
    @pytest.mark.parametrize(
        "matrix",
        [
            pytest.param(R.from_euler('X', np.pi / 5).as_matrix().reshape(1, 3, 3), id="x-rotation"),
            pytest.param(R.from_euler('Y', np.pi / 5).as_matrix().reshape(1, 3, 3), id="y-rotation"),
            pytest.param(R.from_euler('Z', np.pi / 5).as_matrix().reshape(1, 3, 3), id="z-rotation"),
            pytest.param(R.from_euler('XYZ', [0.1, 0.2, 0.3]).as_matrix().reshape(1, 3, 3), id="xyz-rotation"),
            pytest.param(R.from_euler('X', small_angle).as_matrix().reshape(1, 3, 3), id="small-angle-rotation"),
            pytest.param(R.from_euler('Y', np.pi / 2).as_matrix().reshape(1, 3, 3), id="90-degree-rotation"),
        ]
    )
    def test_should_accept_valid_rotation_matrices(self, matrix):
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for valid rotation matrices.")

    # Should handle batches of valid rotation matrices
    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.tile(R.from_euler('X', 0.1).as_matrix(), (5, 1, 1)), id="batch-of-x-rotations"),
            pytest.param(np.tile(R.from_euler('Y', 0.1).as_matrix(), (5, 1, 1)), id="batch-of-y-rotations"),
            pytest.param(np.tile(R.from_euler('Z', 0.1).as_matrix(), (5, 1, 1)), id="batch-of-z-rotations"),
            pytest.param(np.tile(R.from_euler('XYZ', [0.1, 0.2, 0.3]).as_matrix(), (5, 1, 1)), id="batch-of-xyz-rotations"),
        ]
    )
    def test_should_handle_batches_of_valid_rotation_matrices(self, matrices):
        try:
            validate_orthonorm_and_det(matrices)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for a batch of valid rotation matrices.")

    # Should reject non-orthonormal matrices
    def test_should_reject_non_orthonormal_matrices(self):
        matrices = np.ones((1, 3, 3), dtype=np.float64) # not orthonormal
        with pytest.raises(ValueError, match="matrices must be orthonormal rotation matrices"):
            validate_orthonorm_and_det(matrices)

    # Should reject matrices with determinant not equal to +1
    def test_should_reject_non_one_determinant(self):
        matrices = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        matrices[0][0][0] = -1
        with pytest.raises(ValueError, match="matrices must have a determinant of 1"):
            validate_orthonorm_and_det(matrices)

    # Should reject batches containing any invalid matrices
    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.ones((1, 3, 3), dtype=np.float64), id="non-orthonormal"),
            pytest.param(np.zeros((1, 3, 3), dtype=np.float64), id="zero-matrix"),
            pytest.param(np.concatenate(
                (
                    np.tile(R.from_euler('X', 0.1).as_matrix(), (4, 1, 1)),
                    np.ones((1, 3, 3), dtype=np.float64)
                ), axis=0), id="batch-with-one-invalid"
            ),
        ]
    )
    def test_should_reject_batches_with_any_invalid_matrices(self, matrices):
        with pytest.raises(ValueError):
            validate_orthonorm_and_det(matrices)

    # Should handle matrix near singularities without raising false positives
    @pytest.mark.parametrize(
        "angle", 
        [
            0.0 + singularity_tolerance, 
            np.pi - singularity_tolerance
        ]
    )
    def test_should_handle_matrices_near_singularities(self, angle):
        matrix = R.from_euler('Y', angle).as_matrix().reshape(1, 3, 3)
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for a valid rotation matrix near a singularity.")


class TestCreateRotationMatrices:
    # Incorrect shapes should be rejected.
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.zeros(18, dtype=np.float64), id="flat-row"),
            pytest.param(np.zeros((2, 17), dtype=np.float64), id="too-few-columns"),
            pytest.param(np.zeros((2, 19), dtype=np.float64), id="too-many-columns"),
            pytest.param(np.zeros((2, 3, 6), dtype=np.float64), id="3d-array"),
            pytest.param(np.zeros((0, 18), dtype=np.float64), id="empty-array"),
        ],
    )
    def test_should_reject_incorrect_shapes(self, data):
        with pytest.raises(ValueError, match="Data must be a 2D array with exactly 18 columns."):
            create_rotation_matrices(data, "left")

    # Non-float64 inputs should be coerced safely.
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.zeros((2, 18), dtype=np.float32), id="float32"),
            pytest.param(np.zeros((2, 18), dtype=np.int64), id="int64"),
        ],
    )
    def test_should_coerce_non_float64_inputs(self, data):
        matrices = create_rotation_matrices(data, "left")
        assert matrices.dtype == np.float64

    # Invalid arm identifiers should be rejected.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("X", id="string"),
            pytest.param(2, id="dtype"),
            pytest.param(None, id="none"),
        ],
    )
    def test_should_reject_invalid_arm(self, arm):
        data = np.zeros((1, 18), dtype=np.float64)
        with pytest.raises(ValueError, match="arm must be 'left' or 'right'"):
            create_rotation_matrices(data, arm)

    # Left and right arms should be sliced from the correct column block.
    @pytest.mark.parametrize(
        "arm",
        [
            pytest.param("left", id="left"),
            pytest.param("right", id="right"),
        ],
    )
    def test_should_return_the_expected_arm(self, arm):
        left_data = np.full((1, 9), 1.0, dtype=np.float64)
        right_data = np.full((1, 9), 2.0, dtype=np.float64)
        data = np.concatenate((left_data, right_data), axis=1)
        matrix = create_rotation_matrices(data, arm)
        expected = [left_data.reshape(3, 3) if arm == "left" else right_data.reshape(3, 3)]
        assert np.array_equal(matrix, expected)

    # should build the matrix correctly
    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            pytest.param("left", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), id="left"),
            pytest.param("right", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float64), id="right"),
        ],
    )
    def test_should_build_correct_matrix(self, side, expected):
        """Ensure the matrix is built correctly."""
        data = np.arange(1, 19, dtype=np.float64).reshape(1, 18)
        result = create_rotation_matrices(data, side)
        assert np.array_equal(result[0], expected)

    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            pytest.param("left", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), id="left"),
            pytest.param("right", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float64), id="right"),
        ],
    )
    # Should built the matrix correctly for multiple frames
    def test_should_build_correct_matrices_for_multiple_frames(self, side, expected):
        """Ensure the function can handle multiple frames and builds the correct matrices."""
        n_frames = 5
        single_frame = np.arange(1, 19, dtype=np.float64).reshape(1, 18)
        data = np.tile(single_frame, (n_frames, 1))
        result = create_rotation_matrices(data, side)
        for i in range(n_frames):
            assert np.array_equal(result[i], expected)

    # Should return correct dtype and shape
    def test_should_return_correct_dtype_and_shape(self):
        data = np.zeros((5, 18), dtype=np.float64)
        matrices = create_rotation_matrices(data, "left")
        assert matrices.shape == (5, 3, 3)
        assert matrices.dtype == np.float64

    # The function should not mutate the original input values.
    def test_should_not_modify_original_data(self):
        data = np.concatenate(
            [
                np.full((1, 18), 1.0, dtype=np.float64),
                np.full((1, 18), 2.0, dtype=np.float64),
                np.full((1, 18), 3.0, dtype=np.float64)
            ]
        )
        original = data.copy()
        create_rotation_matrices(data, "left")
        assert np.array_equal(data, original)
