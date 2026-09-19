import numpy as np
import pytest
from modules.general_utilities import create_rotation_matrices, convert_heatmap_to_degrees
from schema import Heatmap

# ---- Tests ----


class TestCreateRotationMatrices:
    # Incorrect shapes should be rejected.
    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.zeros(18, dtype=np.float64), id="flat-row"),
            pytest.param(np.zeros((2, 17), dtype=np.float64),
                         id="too-few-columns"),
            pytest.param(np.zeros((2, 19), dtype=np.float64),
                         id="too-many-columns"),
            pytest.param(np.zeros((2, 3, 6), dtype=np.float64), id="3d-array"),
            pytest.param(np.zeros((0, 18), dtype=np.float64),
                         id="empty-array"),
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
        expected = [left_data.reshape(
            3, 3) if arm == "left" else right_data.reshape(3, 3)]
        assert np.array_equal(matrix, expected)

    # should build the matrix correctly
    @pytest.mark.parametrize(
        ("side", "expected"),
        [
            pytest.param("left", np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), id="left"),
            pytest.param("right", np.array([[10, 11, 12], [13, 14, 15], [
                         16, 17, 18]], dtype=np.float64), id="right"),
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
            pytest.param("left", np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), id="left"),
            pytest.param("right", np.array([[10, 11, 12], [13, 14, 15], [
                         16, 17, 18]], dtype=np.float64), id="right"),
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


class TestConvertHeatmapToDegrees:
    @pytest.mark.parametrize(
        "radians, expected",
        [
            (np.array([0.0]), np.array([0.0])),
            (np.array([np.pi / 2]), np.array([90.0])),
            (np.array([np.pi]), np.array([180.0])),
            (np.array([-np.pi / 2]), np.array([-90.0])),
            (np.array([np.pi / 4, np.pi / 2, np.pi]),
             np.array([45.0, 90.0, 180.0])),
        ],
    )
    def test_converts_kinematic_values_to_degrees(self, radians, expected):
        heatmap = Heatmap(
            bin_width=20,
            elevation_range_end=180,
            poe_range_end=360,
            elevation=radians,
            poe=radians,
            ir_er=radians,
            cumulative_motion=radians,
            sample_count=np.array([1, 2, 3]),
        )

        result = convert_heatmap_to_degrees(heatmap)

        np.testing.assert_allclose(result.elevation, expected)
        np.testing.assert_allclose(result.poe, expected)
        np.testing.assert_allclose(result.ir_er, expected)
        np.testing.assert_allclose(result.cumulative_motion, expected)

    def test_preserves_heatmap_metadata(self):
        heatmap = Heatmap(
            bin_width=20,
            elevation_range_end=180,
            poe_range_end=360,
            elevation=np.array([np.pi]),
            poe=np.array([np.pi]),
            ir_er=np.array([np.pi]),
            cumulative_motion=np.array([np.pi]),
            sample_count=np.array([10, 20]),
        )

        result = convert_heatmap_to_degrees(heatmap)

        assert result.bin_width == heatmap.bin_width
        assert result.elevation_range_end == heatmap.elevation_range_end
        assert result.poe_range_end == heatmap.poe_range_end
        assert result.sample_count is heatmap.sample_count

    def test_preserves_array_shape(self):
        values = np.array(
            [
                [0.0, np.pi / 2],
                [np.pi, 3 * np.pi / 2],
            ]
        )

        heatmap = Heatmap(
            bin_width=20,
            elevation_range_end=180,
            poe_range_end=360,
            elevation=values,
            poe=values,
            ir_er=values,
            cumulative_motion=values,
            sample_count=np.ones(values.shape, dtype=np.int32),
        )

        result = convert_heatmap_to_degrees(heatmap)

        assert result.elevation.shape == values.shape
        assert result.poe.shape == values.shape
        assert result.ir_er.shape == values.shape
        assert result.cumulative_motion.shape == values.shape

    def test_does_not_modify_input_heatmap(self):
        values = np.array([0.0, np.pi / 2, np.pi])
        original_values = values.copy()

        heatmap = Heatmap(
            bin_width=20,
            elevation_range_end=180,
            poe_range_end=360,
            elevation=values,
            poe=values.copy(),
            ir_er=values.copy(),
            cumulative_motion=values.copy(),
            sample_count=np.array([1, 2, 3]),
        )

        convert_heatmap_to_degrees(heatmap)

        np.testing.assert_array_equal(heatmap.elevation, original_values)
        np.testing.assert_array_equal(heatmap.poe, original_values)
        np.testing.assert_array_equal(heatmap.ir_er, original_values)
        np.testing.assert_array_equal(heatmap.cumulative_motion, original_values)
