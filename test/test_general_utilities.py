import numpy as np
import pytest
from pathlib import Path
from modules.general_utilities import (
    create_rotation_matrices,
    convert_heatmap_to_degrees,
    convert_all_heatmaps_to_degrees,
    create_rotation_rate_heatmap,
)
from schema import Heatmap, ArmRotationDetails, ParticipantDetails, RotationData

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
        np.testing.assert_array_equal(
            heatmap.cumulative_motion, original_values)


class TestCreateRotationRateHeatmap:
    @staticmethod
    def create_test_heatmap() -> Heatmap:
        values = np.array([[10.0, 20.0], [30.0, 40.0]])
        return Heatmap(
            bin_width=10,
            elevation_range_end=20,
            poe_range_end=20,
            elevation=values.copy(),
            poe=(values + 1).copy(),
            ir_er=(values + 2).copy(),
            cumulative_motion=(values + 3).copy(),
            sample_count=np.array([[1, 2], [3, 4]], dtype=np.int32),
        )

    def test_divides_motion_arrays_by_mocap_duration(self):
        heatmap = self.create_test_heatmap()

        result = create_rotation_rate_heatmap(heatmap, np.float64(2.0))

        np.testing.assert_allclose(result.elevation, heatmap.elevation / 2)
        np.testing.assert_allclose(result.poe, heatmap.poe / 2)
        np.testing.assert_allclose(result.ir_er, heatmap.ir_er / 2)
        np.testing.assert_allclose(
            result.cumulative_motion,
            heatmap.cumulative_motion / 2,
        )

    def test_copies_heatmap_metadata_and_sample_counts(self):
        heatmap = self.create_test_heatmap()

        result = create_rotation_rate_heatmap(heatmap, np.float64(2.0))

        assert result.bin_width == heatmap.bin_width
        assert result.elevation_range_end == heatmap.elevation_range_end
        assert result.poe_range_end == heatmap.poe_range_end
        np.testing.assert_array_equal(
            result.sample_count, heatmap.sample_count)
        assert result.sample_count is not heatmap.sample_count

    def test_does_not_modify_input_heatmap(self):
        heatmap = self.create_test_heatmap()
        original = heatmap.elevation.copy()

        create_rotation_rate_heatmap(heatmap, np.float64(2.0))

        np.testing.assert_array_equal(heatmap.elevation, original)

    def test_rejects_non_positive_mocap_duration(self):
        heatmap = self.create_test_heatmap()

        with pytest.raises(ValueError, match="greater than zero"):
            create_rotation_rate_heatmap(heatmap, np.float64(0.0))


class TestConvertAllHeatmapsToDegrees:
    @staticmethod
    def create_test_heatmap(value: float) -> Heatmap:
        """Create a heatmap with the default 9 x 18 shape."""
        shape = (9, 18)

        return Heatmap(
            elevation=np.full(shape, value, dtype=np.float64),
            poe=np.full(shape, value, dtype=np.float64),
            ir_er=np.full(shape, value, dtype=np.float64),
            cumulative_motion=np.full(shape, value, dtype=np.float64),
            sample_count=np.ones(shape, dtype=np.int32),
        )

    @classmethod
    def create_test_participant(
        cls,
        left_value: float = 0.0,
        right_value: float = 0.0,
    ) -> ParticipantDetails:
        """Create a participant with specified left/right heatmap values."""
        return ParticipantDetails(
            filename=Path("participant_01.txt"),
            rtsa_side=None,
            tsa_side=None,
            dominant_arm=None,
            age=30,
            left=ArmRotationDetails(
                humerothoracic=RotationData(
                    cumulative_rotation_heatmap=cls.create_test_heatmap(
                        left_value)
                )
            ),
            right=ArmRotationDetails(
                humerothoracic=RotationData(
                    cumulative_rotation_heatmap=cls.create_test_heatmap(
                        right_value)
                )
            ),
        )

    @pytest.mark.parametrize(
        "left_radians,right_radians,left_degrees,right_degrees",
        [
            (0.0, 0.0, 0.0, 0.0),
            (np.pi / 2, np.pi, 90.0, 180.0),
            (np.pi, 2 * np.pi, 180.0, 360.0),
            (-np.pi / 2, -np.pi, -90.0, -180.0),
        ],
    )
    def test_converts_left_and_right_heatmaps(
        self,
        left_radians,
        right_radians,
        left_degrees,
        right_degrees,
    ):
        participant = self.create_test_participant(
            left_value=left_radians,
            right_value=right_radians,
        )
        participants = [participant]

        result = convert_all_heatmaps_to_degrees(participants)

        np.testing.assert_allclose(
            result[0].left.humerothoracic.cumulative_rotation_heatmap.elevation,
            left_degrees,
        )
        np.testing.assert_allclose(
            result[0].right.humerothoracic.cumulative_rotation_heatmap.elevation,
            right_degrees,
        )

    def test_converts_all_participants(self):
        participants = [
            self.create_test_participant(
                left_value=np.pi / 2,
                right_value=np.pi,
            ),
            self.create_test_participant(
                left_value=np.pi,
                right_value=np.pi / 4,
            ),
        ]

        result = convert_all_heatmaps_to_degrees(participants)

        expected = [
            (90.0, 180.0),
            (180.0, 45.0),
        ]

        for participant, (left_expected, right_expected) in zip(
            result, expected
        ):
            np.testing.assert_allclose(
                participant.left.humerothoracic.cumulative_rotation_heatmap.elevation,
                left_expected,
            )
            np.testing.assert_allclose(
                participant.right.humerothoracic.cumulative_rotation_heatmap.elevation,
                right_expected,
            )

    def test_converts_all_heatmap_fields(self):
        participant = self.create_test_participant(
            left_value=np.pi / 2,
            right_value=np.pi,
        )

        convert_all_heatmaps_to_degrees([participant])

        left_heatmap = participant.left.humerothoracic.cumulative_rotation_heatmap
        right_heatmap = participant.right.humerothoracic.cumulative_rotation_heatmap

        for heatmap, expected in [
            (left_heatmap, 90.0),
            (right_heatmap, 180.0),
        ]:
            np.testing.assert_allclose(heatmap.elevation, expected)
            np.testing.assert_allclose(heatmap.poe, expected)
            np.testing.assert_allclose(heatmap.ir_er, expected)
            np.testing.assert_allclose(heatmap.cumulative_motion, expected)

    def test_preserves_heatmap_metadata(self):
        participant = self.create_test_participant(
            left_value=np.pi,
            right_value=np.pi / 2,
        )

        left_heatmap = participant.left.humerothoracic.cumulative_rotation_heatmap
        right_heatmap = participant.right.humerothoracic.cumulative_rotation_heatmap

        convert_all_heatmaps_to_degrees([participant])

        for heatmap, original in [
            (participant.left.humerothoracic.cumulative_rotation_heatmap, left_heatmap),
            (participant.right.humerothoracic.cumulative_rotation_heatmap, right_heatmap),
        ]:
            assert heatmap.bin_width == original.bin_width
            assert (
                heatmap.elevation_range_end
                == original.elevation_range_end
            )
            assert heatmap.poe_range_end == original.poe_range_end
            np.testing.assert_array_equal(
                heatmap.sample_count,
                original.sample_count,
            )

    def test_preserves_heatmap_shape(self):
        participant = self.create_test_participant(
            left_value=np.pi,
            right_value=np.pi,
        )

        convert_all_heatmaps_to_degrees([participant])

        for side in ["left", "right"]:
            heatmap = getattr(
                participant,
                side,
            ).humerothoracic.cumulative_rotation_heatmap

            assert heatmap.shape == (9, 18)
            assert heatmap.elevation.shape == (9, 18)
            assert heatmap.poe.shape == (9, 18)
            assert heatmap.ir_er.shape == (9, 18)
            assert heatmap.cumulative_motion.shape == (9, 18)
            assert heatmap.sample_count.shape == (9, 18)

    def test_returns_same_participant_list(self):
        participants = [
            self.create_test_participant(
                left_value=np.pi,
                right_value=np.pi,
            )
        ]

        result = convert_all_heatmaps_to_degrees(participants)

        assert result is participants

    def test_preserves_participant_objects(self):
        participant = self.create_test_participant(
            left_value=np.pi,
            right_value=np.pi,
        )
        participants = [participant]

        result = convert_all_heatmaps_to_degrees(participants)

        assert result[0] is participant

    def test_replaces_heatmaps_with_converted_heatmaps(self):
        participant = self.create_test_participant(
            left_value=np.pi,
            right_value=np.pi / 2,
        )

        original_left = participant.left.humerothoracic.cumulative_rotation_heatmap
        original_right = participant.right.humerothoracic.cumulative_rotation_heatmap

        convert_all_heatmaps_to_degrees([participant])

        assert participant.left.humerothoracic.cumulative_rotation_heatmap is not original_left
        assert participant.right.humerothoracic.cumulative_rotation_heatmap is not original_right
