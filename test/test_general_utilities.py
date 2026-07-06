import numpy as np
import pytest
from modules.general_utilities import (
    create_rotation_matrices,
)


# ---- Global test variables ----
small_angle = 1e-3
singularity_tolerance = 1e-7


# ---- Tests ----
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
