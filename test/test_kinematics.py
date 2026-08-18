import pytest
import numpy as np
from scipy.spatial.transform import Rotation as R
from unittest.mock import MagicMock, patch

from modules.kinematics import (
    PosturalAngles,
    BinBounds,
    BinRotationResult,
    _validate_rotation_data,
    _accumulate_euler_components,
    _get_postural_angles,
    _normalize_postural_angles,
    _compute_incremental_rotation_matrices,
    _create_relative_matrices_and_postural_angles,
    _generate_heatmap_bins,
    _decompose_rotation_matrices_yxy,
    _extract_bin_data,
    _calculate_trace_rotation_angles,
    _calculate_single_bin,
    _add_bin_result_to_heatmap,
    _populate_heatmap,
    calculate_bin_rotations
)
from config import (
    TEST_PRECISION_TOLERANCE,
    SMALLEST_CLINICALLY_RELEVANT_ANGLE as SMALL_ANGLE,
    TEST_SINGULARITY_TOLERANCE
)

from schema import Heatmap

# ----------------------------
# Helper Functions
# ----------------------------
def _is_rotation_matrix(R) -> bool:
    """Sanity check orthonormality."""
    return np.allclose(R.T @ R, np.eye(3), atol=TEST_PRECISION_TOLERANCE)


# ----------------------------
# Global Fixtures
# ----------------------------
@pytest.fixture
def mocker():
    patches = []

    class Mocker:
        def patch(self, target, *args, **kwargs):
            p = patch(target, *args, **kwargs)
            mocked = p.start()
            patches.append(p)
            return mocked

    yield Mocker()

    for p in reversed(patches):
        p.stop()


# ----------------------------
# Tests
# ----------------------------

class TestValidateRotationData:
    @pytest.fixture
    def valid_rotation_data(self) -> np.ndarray:
        """Create valid flattened rotation matrix data."""
        return np.ones((10, 18))

    @pytest.mark.parametrize("arm", ["left", "right"])
    def test_accepts_valid_arm_values(self, valid_rotation_data, arm):
        result = _validate_rotation_data(valid_rotation_data, arm)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 18)

    @pytest.mark.parametrize(
        "invalid_arm",
        [
            "Left",
            "RIGHT",
            "middle",
            "",
            None,
            1,
        ],
    )
    def test_rejects_invalid_arm_values(self, valid_rotation_data, invalid_arm):
        with pytest.raises(ValueError, match="arm must be"):
            _validate_rotation_data(valid_rotation_data, invalid_arm)

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            (18,),          # 1D array
            (10, 17),       # Too few columns
            (10, 19),       # Too many columns
            (2, 3, 3),      # 3D array
            (1, 1, 18),     # Incorrect dimensionality
        ],
    )
    def test_rejects_invalid_data_shapes(self, invalid_shape):
        """Rejects arrays that do not have shape (n_frames, 18)."""
        data = np.ones(invalid_shape)

        with pytest.raises(ValueError):
            _validate_rotation_data(data, "left")

    def test_rejects_empty_array(self):
        """Rejects arrays with zero frames."""
        data = np.empty((0, 18))

        with pytest.raises(
            ValueError
        ):
            _validate_rotation_data(data, "left")

    def test_converts_input_to_float64(self):
        """Converts integer input data to np.float64."""
        data = np.ones((5, 18), dtype=np.int32)

        result = _validate_rotation_data(data, "right") # type: ignore[arg-type]

        assert result.dtype == np.float64

    def test_output_is_numpy_array(self):
        """Returns a numpy array."""
        data =np.ones((5, 18), dtype=np.float64)
        result = _validate_rotation_data(data, "left")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_preserves_valid_input_values(
        self,
        valid_rotation_data,
    ):
        """Does not modify valid input values."""
        data = valid_rotation_data.copy()
        data[0, 0] = 42.5

        result = _validate_rotation_data(data, "left")

        np.testing.assert_array_equal(result, data)

    def test_accepts_single_frame(self):
        """Accepts data containing one frame."""
        data = np.zeros((1, 18))

        result = _validate_rotation_data(data, "right")

        assert result.shape == (1, 18)
        assert result.dtype == np.float64


class TestGetPosturalAngles:
    @pytest.mark.parametrize(
        "rotation_matrices, expected_angles",
        [
            (
                # Identity rotation
                np.array(
                    [
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    ]
                ),
                PosturalAngles(
                    poe=np.array([0.0], dtype=np.float64),
                    elevation=np.array([0.0], dtype=np.float64),
                    ir_er=np.array([0.0], dtype=np.float64),
                ),
            ),
            (
                # Pure 90 degree rotation about X
                R.from_euler(
                    "X",
                    90,
                    degrees=True,
                ).as_matrix()[None, :, :],
                PosturalAngles(
                    poe=np.array([0.0]),
                    elevation=np.array([90.0]),
                    ir_er=np.array([0.0]),
                ),
            ),
            (
                # Pure 90 degree rotation about Y
                R.from_euler(
                    "Y",
                    90,
                    degrees=True,
                ).as_matrix()[None, :, :],
                PosturalAngles(
                    poe=np.array([90.0]),
                    elevation=np.array([0.0]),
                    ir_er=np.array([0.0]),
                ),
            ),
            (
                # Multiple frames with known YXY rotations
                R.from_euler(
                    "YXY",
                    [
                        [30, 45, 60],
                        [90, 20, -30],
                    ],
                    degrees=True,
                ).as_matrix(),
                PosturalAngles(
                    poe=np.array([30.0, 90.0]),
                    elevation=np.array([45.0, 20.0]),
                    ir_er=np.array([60.0, -30.0]),
                ),
            ),
        ],
    )
    def test_get_position_angles(self, rotation_matrices, expected_angles):
        """Test Euler angle decomposition from rotation matrices."""

        actual_angles = _get_postural_angles(rotation_matrices)
        # appeasing the type checker since the dataclass allows None values
        assert actual_angles.poe is not None
        assert actual_angles.elevation is not None
        assert actual_angles.ir_er is not None

        np.testing.assert_allclose(
            actual_angles.poe,
            expected_angles.poe,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            actual_angles.elevation,
            expected_angles.elevation,
            atol=1e-10,
        )

        np.testing.assert_allclose(
            actual_angles.ir_er,
            expected_angles.ir_er,
            atol=1e-10,
        )


class TestNormalizePosturalAngles:
    @pytest.mark.parametrize(
        ("raw_angles, expected_angles"),
        [
            (
                PosturalAngles(
                    poe=np.array([-1.0, 361.0, 725.0]),
                    elevation=np.array([-45.0, 45.0, -120.0]),
                    ir_er=np.array([-1.0, 361.0, 721.0]),
                ),
                PosturalAngles(
                    poe=np.array([359.0, 1.0, 5.0]),
                    elevation=np.array([45.0, 45.0, 120.0]),
                    ir_er=np.array([359.0, 1.0, 1.0]),
                ),
            ),
            (
                PosturalAngles(
                    poe=np.array([0.0]),
                    elevation=np.array([250.0]),
                    ir_er=np.array([0.0]),
                ),
                PosturalAngles(
                    poe=np.array([0.0]),
                    elevation=np.array([70.0]),
                    ir_er=np.array([0.0]),
                ),
            ),
        ],
    )
    def test_normalize_position_angles(self, raw_angles, expected_angles):
        """Test position angle normalization for heatmap binning."""

        normalized_angles = _normalize_postural_angles(raw_angles)

        # Assert no NoneTypes to appease the type checker
        assert normalized_angles.poe is not None
        assert normalized_angles.elevation is not None
        assert normalized_angles.ir_er is not None

        np.testing.assert_array_equal(
            normalized_angles.poe,
            expected_angles.poe,
        )

        np.testing.assert_array_equal(
            normalized_angles.elevation,
            expected_angles.elevation,
        )

        np.testing.assert_array_equal(
            normalized_angles.ir_er,
            expected_angles.ir_er,
        )

        # Verify binning constraints
        assert np.all(normalized_angles.poe >= 0)
        assert np.all(normalized_angles.poe < 360)

        assert np.all(normalized_angles.ir_er >= 0)
        assert np.all(normalized_angles.ir_er < 360)

        assert np.all(normalized_angles.elevation >= 0)
        assert np.all(normalized_angles.elevation <= 180)


class TestComputeIncrementalRotationMatrices:
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64),
                         id="2x2-matrices"),
            pytest.param(np.zeros((3, 3, 4), dtype=np.float64),
                         id="wrong-last-dim"),
            pytest.param(np.zeros((4, 9), dtype=np.float64),
                         id="flattened-rows"),
        ],
    )
    def test_should_reject_non_3x3_inputs(self, rotation_matrices):
        """The function should only accept batches of 3x3 matrices."""
        with pytest.raises(ValueError):
            _compute_incremental_rotation_matrices(rotation_matrices)

    # Test that fewer than two frames are rejected with a ValueError.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros((0, 3, 3), dtype=np.float64),
                         id="zero-frames"),
            pytest.param(
                np.stack([np.eye(3, dtype=np.float64)]), id="single-frame"),
        ],
    )
    def test_should_reject_insufficient_frames(self, rotation_matrices):
        """The function should require at least two frames so a relative rotation exists."""
        with pytest.raises(ValueError):
            _compute_incremental_rotation_matrices(rotation_matrices)

    # Test that non-float64 inputs are accepted and safely coerced to float64.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros((2, 3, 3), dtype=np.float32), id="float32"),
            pytest.param(np.zeros((2, 3, 3), dtype=np.int64), id="int64"),
        ],
    )
    def test_should_coerce_non_float64_inputs(self, rotation_matrices):
        """The function should coerce input matrices to float64."""
        deltas = _compute_incremental_rotation_matrices(rotation_matrices)
        assert deltas.dtype == np.float64

    # Test that a sequence of identical matrices produces identity deltas.
    @pytest.mark.parametrize("n_frames", [2, 5, 10])
    def test_should_return_identity_deltas_for_constant_sequence(self, n_frames):
        """A constant absolute-orientation sequence should produce identity deltas."""
        frames = np.stack([np.eye(3, dtype=np.float64)
                          for _ in range(n_frames)])
        deltas = _compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (n_frames - 1, 3, 3)
        for D in deltas:
            assert np.allclose(D, np.eye(3), atol=TEST_PRECISION_TOLERANCE)

    # Test that the function correctly computes the relative rotation.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X", np.pi / 6, id="x-axis"),
            pytest.param(R.from_euler, "Y", np.pi / 4, id="y-axis"),
            pytest.param(R.from_euler, "Z", np.pi / 3, id="z-axis"),
            pytest.param(R.from_euler, "XYZ", [
                         0.2, 0.2, 0.2], id="combined-xyz"),
        ],
    )
    def test_should_return_expected_relative_matrix(self, rotation_builder, sequence, angle):
        """For a two-frame sequence, the returned delta should equal R_current @ R_previous.T."""
        R0 = np.eye(3, dtype=np.float64)
        R1 = rotation_builder(sequence, angle).as_matrix()
        frames = np.stack([R0, R1])
        deltas = _compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (1, 3, 3)
        expected = R1 @ R0.T
        assert np.allclose(deltas[0], expected, atol=TEST_PRECISION_TOLERANCE)

    # Test that the cumulative product of deltas reconstructs the original sequence.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle", "n_steps"),
        [
            pytest.param(R.from_euler, "X", np.pi/12, 4, id="small-x"),
            pytest.param(R.from_euler, "Y", np.pi/12, 4, id="small-y"),
            pytest.param(R.from_euler, "Z", np.pi/12, 4, id="small-z"),
            pytest.param(R.from_euler, "YXY", [
                         0.15, 0.15, 0.15], 4, id="small-yxy"),
        ],
    )
    def test_should_reconstruct_to_absolute_orientation(self, rotation_builder, sequence, angle, n_steps):
        """The cumulative product of deltas should reproduce the original sequence."""
        D = rotation_builder(sequence, angle).as_matrix()
        frames = [np.eye(3, dtype=np.float64)]
        for _ in range(n_steps - 1):
            frames.append(frames[-1] @ D)
        frames = np.stack(frames)

        deltas = _compute_incremental_rotation_matrices(frames)

        # Reconstruct sequential frames from R0 and deltas
        reconstructed = [frames[0]]
        for i in range(deltas.shape[0]):
            reconstructed.append(reconstructed[-1] @ deltas[i])
        reconstructed = np.stack(reconstructed)

        assert np.allclose(reconstructed, frames,
                           atol=TEST_PRECISION_TOLERANCE)

    # Test that each delta is a valid rotation matrix (orthonormal with determinant 1).
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X",  np.pi / 8, id="x-axis"),
            pytest.param(R.from_euler, "Y", np.pi / 7, id="y-axis"),
            pytest.param(R.from_euler, "Z", np.pi / 9, id="z-axis"),
            pytest.param(R.from_euler, "YXY", [0.1, 0.2, 0.3], id="combined"),
        ],
    )
    def test_should_return_valid_rotation_matrices(self, rotation_builder, sequence, angle):
        """Each relative matrix should still be a valid rotation matrix."""
        n_steps = 5
        D = rotation_builder(sequence, angle).as_matrix()
        frames = [np.eye(3, dtype=np.float64)]

        for _ in range(n_steps - 1):
            frames.append(frames[-1] @ D)
        frames = np.stack(frames)

        deltas = _compute_incremental_rotation_matrices(frames)
        for D in deltas:
            assert _is_rotation_matrix(D)

    # Test that the function handles small and large rotations appropriately.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X",  SMALL_ANGLE, id="small-x"),
            pytest.param(R.from_euler, "Y", SMALL_ANGLE, id="small-y"),
            pytest.param(R.from_euler, "Z", SMALL_ANGLE, id="small-z"),
            pytest.param(R.from_euler, "YXY", [
                         SMALL_ANGLE, SMALL_ANGLE, SMALL_ANGLE], id="small-yxy"),
            pytest.param(R.from_euler, "X",  np.pi / 2, id="large-x"),
            pytest.param(R.from_euler, "Y", np.pi * 0.75, id="large-y"),
            pytest.param(R.from_euler, "Z", np.pi * 0.9, id="large-z"),
            pytest.param(R.from_euler, "YXY", [
                         np.pi / 6, np.pi / 6, np.pi / 6], id="large-yxy"),
        ]
    )
    def test_should_handle_small_and_large_rotations(self, rotation_builder, sequence, angle):
        """Small absolute changes should yield small deltas, and larger changes should yield larger deltas."""
        R0 = np.eye(3, dtype=np.float64)
        R1 = rotation_builder(sequence, angle).as_matrix()
        deltas = _compute_incremental_rotation_matrices(np.stack([R0, R1]))
        # Compute rotation angle from a rotation matrix using the trace formula
        # theta = arccos((trace(R) - 1) / 2)
        delta_trace = np.trace(deltas[0])
        delta_angle = np.arccos(np.clip((delta_trace - 1) / 2, -1.0, 1.0))

        r1_trace = np.trace(R1)
        expected_angle = np.arccos(np.clip((r1_trace - 1) / 2, -1.0, 1.0))
        assert delta_angle == pytest.approx(expected_angle)

    # Test that the function returns the correct shape and dtype.
    @pytest.mark.parametrize("n_frames", [2, 3, 8])
    def test_should_return_expected_shape_and_dtype(self, n_frames):
        """The function should return (n_frames - 1, 3, 3) float64 matrices."""
        D = R.from_euler("X", 0.1).as_matrix()
        frames = [np.eye(3, dtype=np.float64)]
        for _ in range(n_frames - 1):
            frames.append(frames[-1] @ D)
        frames = np.stack(frames)

        deltas = _compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (n_frames - 1, 3, 3)
        assert deltas.dtype == np.float64


class TestCreateRelativeMatricesAndPosturalAngles:
    """Tests for _create_relative_matrices_and_postural_angles."""

    @pytest.fixture
    def data(self) -> np.ndarray:
        """Create valid input data."""
        rng = np.random.default_rng(42)
        return rng.random((10, 18))

    @pytest.fixture
    def matrices(self) -> np.ndarray:
        """Mock rotation matrices."""
        rng = np.random.default_rng(1)
        return rng.random((10, 3, 3))

    @pytest.fixture
    def postural_angles(self) -> MagicMock:
        """Mock PosturalAngles object."""
        return MagicMock(spec=PosturalAngles)

    @pytest.fixture
    def normalized_postural_angles(self) -> MagicMock:
        """Mock normalized PosturalAngles object."""
        return MagicMock(spec=PosturalAngles)

    @pytest.fixture
    def relative_matrices(self) -> np.ndarray:
        """Mock relative rotation matrices."""
        rng = np.random.default_rng(2)
        return rng.random((10, 3, 3))

    def test_calls_dependencies_once_and_returns_expected_values(
        self,
        mocker,
        data,
        matrices,
        postural_angles,
        normalized_postural_angles,
        relative_matrices,
    ):
        """Calls each dependency exactly once and returns their outputs."""
        create = mocker.patch(
            "modules.kinematics.create_rotation_matrices",
            return_value=matrices,
        )
        get_angles = mocker.patch(
            "modules.kinematics._get_postural_angles",
            return_value=postural_angles,
        )
        normalize = mocker.patch(
            "modules.kinematics._normalize_postural_angles",
            return_value=normalized_postural_angles,
        )
        compute = mocker.patch(
            "modules.kinematics._compute_incremental_rotation_matrices",
            return_value=relative_matrices,
        )

        result_matrices, result_angles = _create_relative_matrices_and_postural_angles(
            data,
            "left",
        )

        create.assert_called_once_with(data, "left")
        get_angles.assert_called_once_with(matrices)
        normalize.assert_called_once_with(postural_angles)
        compute.assert_called_once_with(matrices)

        assert result_matrices is relative_matrices
        assert result_angles is normalized_postural_angles

    def test_does_not_modify_input_data(
        self,
        mocker,
        data,
        matrices,
        postural_angles,
        normalized_postural_angles,
        relative_matrices,
    ) -> None:
        """Does not modify the input data."""
        original = data.copy()

        mocker.patch(
            "modules.kinematics.create_rotation_matrices",
            return_value=matrices,
        )
        mocker.patch(
            "modules.kinematics._get_postural_angles",
            return_value=postural_angles,
        )
        mocker.patch(
            "modules.kinematics._normalize_postural_angles",
            return_value=normalized_postural_angles,
        )
        mocker.patch(
            "modules.kinematics._compute_incremental_rotation_matrices",
            return_value=relative_matrices,
        )

        _create_relative_matrices_and_postural_angles(data, "left")

        np.testing.assert_array_equal(data, original)


class TestGenerateHeatmapBins:
    @pytest.mark.parametrize(
        "bin_width, elevation_end, poe_end, expected_count",
        [
            (10, 180, 360, 18 * 36),
            (20, 180, 360, 9 * 18),
            (30, 180, 360, 6 * 12),
            (90, 180, 360, 2 * 4),
        ],
    )
    def test_generates_expected_number_of_bins(
        self,
        bin_width: int,
        elevation_end: int,
        poe_end: int,
        expected_count: int,
    ):
        bins = list(
            _generate_heatmap_bins(
                bin_width,
                elevation_end,
                poe_end,
            )
        )

        assert len(bins) == expected_count

    def test_generates_expected_bin_boundaries(self) -> None:
        """Generates the correct sequence of bin boundaries."""
        bins = list(
            _generate_heatmap_bins(
                bin_width=90,
                elevation_range_end=180,
                poe_range_end=180,
            )
        )

        expected = [
            BinBounds(elevation_start=0, elevation_end=90, poe_start=0, poe_end=90),
            BinBounds(elevation_start=0, elevation_end=90, poe_start=90, poe_end=180),
            BinBounds(elevation_start=90, elevation_end=180, poe_start=0, poe_end=90),
            BinBounds(elevation_start=90, elevation_end=180, poe_start=90, poe_end=180),
        ]

        assert bins == expected

    @pytest.mark.parametrize(
        "bin_width, elevation_end, poe_end",
        [
            (10, 180, 360),
            (20, 180, 360),
            (45, 180, 360),
        ],
    )
    def test_returns_binbounds_with_integer_fields(
        self,
        bin_width: int,
        elevation_end: int,
        poe_end: int,
    ):
        bins = list(
            _generate_heatmap_bins(
                bin_width,
                elevation_end,
                poe_end,
            )
        )

        for bin_ in bins:
            bin_obj = BinBounds(**bin_.__dict__)
            assert isinstance(bin_obj.elevation_start, int)
            assert isinstance(bin_obj.elevation_end, int)
            assert isinstance(bin_obj.poe_start, int)
            assert isinstance(bin_obj.poe_end, int)

    def test_bin_width_is_consistent(self):
        """Each generated bin has the requested width."""
        bin_width = 30

        bins = list(
            _generate_heatmap_bins(
                bin_width,
                elevation_range_end=180,
                poe_range_end=360,
            )
        )

        for bin_ in bins:
            assert (
                bin_.elevation_end - bin_.elevation_start
                == bin_width
            )
            assert (
                bin_.poe_end - bin_.poe_start
                == bin_width
            )

    def test_returns_iterator(self):
        iterator = _generate_heatmap_bins(
            bin_width=30,
            elevation_range_end=180,
            poe_range_end=360,
        )

        assert iter(iterator) is iterator


class TestExtractBinData:

    @pytest.fixture
    def relative_rotations(self) -> np.ndarray:
        """
        Create mock relative rotation matrices.

        Each matrix is unique so it is obvious which transitions were selected.
        """
        return np.array(
            [
                np.eye(3) * 10,  # transition 0 -> 1
                np.eye(3) * 20,  # transition 1 -> 2
                np.eye(3) * 30,  # transition 2 -> 3
                np.eye(3) * 40,  # transition 3 -> 4
            ],
            dtype=np.float64,
        )

    @pytest.fixture
    def postural_data(self) -> PosturalAngles:
        """
        Postural data representing the starting position of each relative rotation.
        """
        return PosturalAngles(
            poe=np.array([10, 20, 30, 40, 50], dtype=np.float64),
            elevation=np.array([10, 20, 30, 40, 50], dtype=np.float64),
            ir_er=np.array([10, 20, 30, 40, 50], dtype=np.float64),
        )

    def test_returns_only_frames_inside_requested_bin(
        self,
        relative_rotations,
        postural_data,
    ):
        """Returns relative rotations where both elevation and POE fall inside the bin."""

        result = _extract_bin_data(
            data=relative_rotations,
            postural_data=postural_data,
            elevation_start=15,
            elevation_end=45,
            poe_start=15,
            poe_end=45,
        )

        expected = np.array(
            [
                np.eye(3) * 20,
                np.eye(3) * 30,
                np.eye(3) * 40,
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(result, expected)

    def test_lower_bounds_are_inclusive_and_upper_bounds_are_exclusive(
        self,
        relative_rotations,
        postural_data,
    ):
        """Frames exactly at the lower bound are included, and frames exactly at the upper bound are excluded."""

        result = _extract_bin_data(
            relative_rotations,
            postural_data,
            elevation_start=20,
            elevation_end=40,
            poe_start=20,
            poe_end=40,
        )

        expected = np.array(
            [
                np.eye(3) * 20,
                np.eye(3) * 30,
            ],
            dtype=np.float64,
        )

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "elevation_start,elevation_end,poe_start,poe_end",
        [
            (20, 20, 0, 10),
            (30, 20, 0, 10),
            (0, 10, 10, 10),
            (0, 10, 20, 10),
        ],
    )
    def test_invalid_bin_bounds_raise_value_error(
        self,
        relative_rotations,
        postural_data,
        elevation_start,
        elevation_end,
        poe_start,
        poe_end,
    ):
        with pytest.raises(ValueError):
            _extract_bin_data(
                relative_rotations,
                postural_data,
                elevation_start,
                elevation_end,
                poe_start,
                poe_end,
            )

    def test_returns_empty_array_when_no_frames_match(
        self,
        relative_rotations,
        postural_data,
    ):
        result = _extract_bin_data(
            relative_rotations,
            postural_data,
            elevation_start=100,
            elevation_end=120,
            poe_start=100,
            poe_end=120,
        )

        assert result.shape == (0, 3, 3)

    def test_does_not_modify_original_data(
        self,
        relative_rotations,
        postural_data,
    ):
        original = relative_rotations.copy()

        _extract_bin_data(
            relative_rotations,
            postural_data,
            elevation_start=0,
            elevation_end=90,
            poe_start=0,
            poe_end=90,
        )

        np.testing.assert_array_equal(relative_rotations, original)


class TestDecomposeRotationMatricesYXY:
    # Test that non-3x3 inputs are rejected with a ValueError.
    @pytest.mark.parametrize(
        "relative_rotations",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64),
                         id="2x2-matrices"),
            pytest.param(np.zeros((3, 3, 4), dtype=np.float64),
                         id="wrong-last-dim"),
            pytest.param(np.zeros((4, 9), dtype=np.float64),
                         id="flattened-rows"),
        ],
    )
    def test_should_reject_non_3x3_inputs(self, relative_rotations):
        with pytest.raises(ValueError):
            _decompose_rotation_matrices_yxy(relative_rotations)

    # Test that empty batch input raises a ValueError.
    @pytest.mark.parametrize(
        "relative_rotations",
        [
            pytest.param(
                np.stack([np.eye(3, dtype=np.float32)]), id="float32"),
            pytest.param(np.stack([np.eye(3, dtype=np.int64)]), id="int64"),
        ],
    )
    def test_should_coerce_non_float64_inputs(self, relative_rotations):
        angles = _decompose_rotation_matrices_yxy(relative_rotations)
        assert angles.dtype == np.float64

    # Test that empty batch input raises a ValueError.
    def test_should_reject_empty_batch(self):
        empty = np.zeros((0, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            _decompose_rotation_matrices_yxy(empty)

    # Output shape and dtype for single and multiple steps
    @pytest.mark.parametrize("n_steps", [1, 5])
    def test_should_return_expected_shape_and_dtype(self, n_steps):
        matrices = np.stack([
            R.from_euler("YXY", [0.1, 0.05, 0.2]).as_matrix()
            for _ in range(n_steps)
        ])
        angles = _decompose_rotation_matrices_yxy(matrices)
        assert angles.shape == (n_steps, 3)
        assert angles.dtype == np.float64

    # Identity matrices should yield zero angles
    def test_identity_should_decompose_to_zeros(self):
        matrices = np.stack([np.eye(3, dtype=np.float64) for _ in range(3)])
        angles = _decompose_rotation_matrices_yxy(matrices)
        assert np.allclose(angles, np.zeros((3, 3), dtype=np.float64))

    # Recomposition: decompose matrices built as Ry(a) @ Rx(b) @ Ry(c)
    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [
            pytest.param(0.1, 0.05, -0.2, id="small-mix"),
            pytest.param(np.pi / 6, 0.2, np.pi / 8, id="medium-mix"),
            pytest.param(-0.3, np.pi / 3, 0.4, id="mixed-signs"),
        ],
    )
    def test_known_yxy_compositions_reconstruct(self, a, b, c):
        matrices = np.stack([R.from_euler("YXY", [a, b, c]).as_matrix()])
        angles = _decompose_rotation_matrices_yxy(matrices)[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(
            recomposed, matrices[0], atol=TEST_PRECISION_TOLERANCE)

    # Singularity: middle angle (X) near zero should still reconstruct
    @pytest.mark.parametrize("beta", [0.0, 1e-8])
    def test_singularity_beta_near_zero(self, beta):
        a, c = 0.3, -0.4
        M = R.from_euler("YXY", [a, beta, c]).as_matrix()
        angles = _decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=TEST_SINGULARITY_TOLERANCE)

    # Singularity: middle angle (X) near pi should still reconstruct
    def test_singularity_beta_near_pi(self):
        beta = np.pi - 1e-8
        a, c = 0.2, 0.5
        M = R.from_euler("YXY", [a, beta, c]).as_matrix()
        angles = _decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=TEST_SINGULARITY_TOLERANCE)

    # Clipping robustness: tiny noise pushing values slightly outside [-1,1]
    def test_clipping_robustness(self):
        a, b, c = 0.4, 0.9, -0.2
        M = R.from_euler("YXY", [a, b, c]).as_matrix()
        noisy = M.copy()
        noisy += np.random.default_rng(1).normal(scale=1e-12, size=M.shape)
        angles = _decompose_rotation_matrices_yxy(np.stack([noisy]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=TEST_PRECISION_TOLERANCE)

    # Non-rotation matrices should raise ValueError
    def test_should_reject_non_rotation_matrices(self):
        bad = np.eye(3, dtype=np.float64)
        bad[0] *= 2.0  # break orthonormality
        with pytest.raises(ValueError):
            _decompose_rotation_matrices_yxy(np.stack([bad]))

    # Determinism: repeated calls return identical results
    def test_deterministic_outputs(self):
        # Use the same matrix for consistency
        M = R.from_euler("YXY", [0.25, 0.15, -0.35]).as_matrix()
        first = _decompose_rotation_matrices_yxy(np.stack([M]))
        second = _decompose_rotation_matrices_yxy(np.stack([M]))
        assert np.allclose(first, second)

    # Sensitivity across magnitudes: tiny and near-pi angles reconstruct
    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [
            pytest.param(SMALL_ANGLE, SMALL_ANGLE, -
                         SMALL_ANGLE, id="tiny-angles"),
            pytest.param(1.2, np.pi - SMALL_ANGLE, -0.9, id="large-middle"),
        ],
    )
    def test_small_and_large_angle_sensitivity(self, a, b, c):
        M = R.from_euler("YXY", [a, b, c]).as_matrix()
        angles = _decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=TEST_PRECISION_TOLERANCE)


class TestAccumulateEulerComponents:
    @pytest.mark.parametrize(
        "euler_angles, expected",
        [
            (
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                    ]
                ),
                (12.0, 15.0, 18.0),
            ),
            (
                np.array(
                    [
                        [10.5, 20.5, 30.5],
                    ]
                ),
                (10.5, 20.5, 30.5),
            ),
            (
                np.zeros((4, 3)),
                (0.0, 0.0, 0.0),
            ),
        ],
    )
    def test_returns_componentwise_sums(self, euler_angles, expected):
        """Returns the cumulative sum of each Euler component."""
        result = _accumulate_euler_components(euler_angles)

        assert result == pytest.approx(expected)

    def test_negative_values_are_summed_by_absolute_value(self):
        """Negative values are converted to absolute values before summing."""

        euler_angles = np.array(
            [
                [-1.0, 2.0, -3.0],
                [4.0, -5.0, 6.0],
            ]
        )

        expected = (5.0, 7.0, 9.0)
        result = _accumulate_euler_components(euler_angles)

        assert result == expected

    def test_integer_input_still_outputs_float_values(self):
        """Integer arrays are accepted and correctly summed."""

        euler_angles = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ],
            dtype=np.int32,
        )
        expected = (5.0, 7.0, 9.0)

        result = _accumulate_euler_components(
            euler_angles)  # type: ignore[arg-type]

        assert result == pytest.approx((5.0, 7.0, 9.0))

    @pytest.mark.parametrize(
        "invalid_shape",
        [
            np.array([]),
            np.array([1.0, 2.0, 3.0]),          # 1D
            np.array([[1.0, 2.0]]),             # 2 columns
            np.array([[1.0, 2.0, 3.0, 4.0]]),   # 4 columns
            np.zeros((2, 2, 3)),                # 3D
        ],
    )
    def test_raises_for_invalid_shape(self, invalid_shape):
        """Raises ValueError when input is not shaped (n_steps, 3)."""

        with pytest.raises(
            ValueError,
            match="euler_angles must have shape \\(n_steps, 3\\)",
        ):
            _accumulate_euler_components(invalid_shape)

    def test_raises_for_empty_input(self):
        """Raises ValueError when no rows are provided."""

        euler_angles = np.empty((0, 3), dtype=np.float64)

        with pytest.raises(
            ValueError,
            match="euler_angles must contain at least one row",
        ):
            _accumulate_euler_components(euler_angles)


class TestCalculateSingleBin:
    @pytest.fixture
    def postural_angles(self) -> MagicMock:
        """Mock PosturalAngles."""
        return MagicMock(spec=PosturalAngles)

    @pytest.fixture
    def euler_angles(self) -> np.ndarray:
        """Mock decomposed Euler angles."""
        rng = np.random.default_rng(2)
        return rng.random((5, 3))

    def test_calls_dependencies_once_and_returns_expected_result(
        self,
        mocker,
        postural_angles,
        euler_angles,
    ):
        trace_values = np.array([2.0, 3.5, 4.5], dtype=np.float64)
        euler_bin = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )

        mock_extract_bin_data = mocker.patch(
            "modules.kinematics._extract_bin_data",
            side_effect=[euler_bin, trace_values],
        )
        mock_accumulate_euler = mocker.patch(
            "modules.kinematics._accumulate_euler_components",
            return_value=(
                np.float64(10.0),
                np.float64(20.0),
                np.float64(30.0),
            ),
        )

        expected_result = BinRotationResult(
            elevation=np.float64(10.0),
            poe=np.float64(20.0),
            ir_er=np.float64(30.0),
            cumulative_motion=np.float64(np.sum(trace_values)),
            sample_count=euler_bin.shape[0] 
        )

        result = _calculate_single_bin(
            traces=trace_values,
            euler_components=euler_angles,
            postural_angles=postural_angles,
            elevation_start=0,
            elevation_end=20,
            poe_start=40,
            poe_end=60,
        )

        assert mock_extract_bin_data.call_count == 2

        first_call = mock_extract_bin_data.call_args_list[0].kwargs
        second_call = mock_extract_bin_data.call_args_list[1].kwargs

        np.testing.assert_array_equal(first_call["data"], euler_angles)
        np.testing.assert_array_equal(second_call["data"], trace_values)

        assert first_call["postural_data"] is postural_angles
        assert second_call["postural_data"] is postural_angles
        assert first_call["elevation_start"] == 0
        assert second_call["elevation_start"] == 0
        assert first_call["elevation_end"] == 20
        assert second_call["elevation_end"] == 20
        assert first_call["poe_start"] == 40
        assert second_call["poe_start"] == 40
        assert first_call["poe_end"] == 60
        assert second_call["poe_end"] == 60

        mock_accumulate_euler.assert_called_once_with(euler_bin)

        assert isinstance(result, BinRotationResult)
        assert result == expected_result

    def test_returns_zero_result_when_bin_is_empty(
        self,
        mocker,
        postural_angles,
    ):
        """Returns zeros and skips further processing when no data is found."""
        empty_euler_bin = np.empty((0, 3), dtype=np.float64)
        empty_trace_bin = np.empty((0,), dtype=np.float64)

        mock_extract_bin_data = mocker.patch(
            "modules.kinematics._extract_bin_data",
            side_effect=[empty_euler_bin, empty_trace_bin],
        )
        mock_accumulate_euler = mocker.patch(
            "modules.kinematics._accumulate_euler_components",
        )

        expected_result = BinRotationResult(
            elevation=np.float64(0),
            poe=np.float64(0),
            ir_er=np.float64(0),
            cumulative_motion=np.float64(0),
            sample_count=0,
        )
        result = _calculate_single_bin(
            traces=np.empty((0,), dtype=np.float64),
            euler_components=np.empty((0, 3), dtype=np.float64),
            postural_angles=postural_angles,
            elevation_start=0,
            elevation_end=20,
            poe_start=40,
            poe_end=60,
        )

        assert mock_extract_bin_data.call_count == 2
        mock_accumulate_euler.assert_not_called()

        assert isinstance(result, BinRotationResult)
        assert result == expected_result


class TestAddBinResultToHeatmap:
    """Tests for _add_bin_result_to_heatmap."""

    @pytest.fixture
    def heatmap(self) -> Heatmap:
        """Create a heatmap with existing 2D data."""
        return Heatmap(
            bin_width=90,
            elevation_range_end=180,
            poe_range_end=180,
            elevation=np.array([[1.0, 2.0], [3.0, 4.0]]),
            poe=np.array([[5.0, 6.0], [7.0, 8.0]]),
            ir_er=np.array([[9.0, 10.0], [11.0, 12.0]]),
            cumulative_motion=np.array([[15.0, 18.0], [21.0, 24.0]]),
            sample_count=np.array([[30, 40], [50, 60]], dtype=np.int32),
        )

    @pytest.fixture
    def bin_result(self) -> BinRotationResult:
        """Create a single bin result."""
        return BinRotationResult(
            elevation=np.float64(10.0),
            poe=np.float64(20.0),
            ir_er=np.float64(30.0),
            cumulative_motion=np.float64(60.0),
            sample_count=5,
        )

    def test_writes_bin_result_data_to_expected_heatmap_cell(
        self,
        heatmap: Heatmap,
        bin_result: BinRotationResult,
    ):
        original_heatmap = heatmap
        result = _add_bin_result_to_heatmap(
            heatmap,
            bin_result,
            elevation_index=1,
            poe_index=0,
        )

        assert result is None

        # Verify in-place modification
        assert heatmap is original_heatmap

        np.testing.assert_array_equal(
            heatmap.elevation,
            np.array([[1.0, 2.0], [10.0, 4.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.poe,
            np.array([[5.0, 6.0], [20.0, 8.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.ir_er,
            np.array([[9.0, 10.0], [30.0, 12.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.cumulative_motion,
            np.array([[15.0, 18.0], [60.0, 24.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.sample_count,
            np.array([[30, 40], [5, 60]], dtype=np.int32),
        )

    def test_writes_to_preallocated_empty_heatmap(
        self,
        bin_result: BinRotationResult,
    ):
        """Writes the first bin result into the specified cell."""
        heatmap = Heatmap(
            bin_width=90,
            elevation_range_end=180,
            poe_range_end=180,
            elevation=np.zeros((2, 2), dtype=np.float64),
            poe=np.zeros((2, 2), dtype=np.float64),
            ir_er=np.zeros((2, 2), dtype=np.float64),
            cumulative_motion=np.zeros((2, 2), dtype=np.float64),
            sample_count=np.zeros((2, 2), dtype=np.int32),
        )

        _add_bin_result_to_heatmap(
            heatmap,
            bin_result,
            elevation_index=0,
            poe_index=1,
        )

        np.testing.assert_array_equal(
            heatmap.elevation,
            np.array([[0.0, 10.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.poe,
            np.array([[0.0, 20.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.ir_er,
            np.array([[0.0, 30.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.cumulative_motion,
            np.array([[0.0, 60.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            heatmap.sample_count,
            np.array([[0, 5], [0, 0]], dtype=np.int32),
        )


class TestPopulateHeatmap:
    """Tests for _populate_heatmap."""

    @pytest.fixture
    def heatmap(self) -> Heatmap:
        """Create a test heatmap."""
        return Heatmap(
            bin_width=30,
            elevation_range_end=180,
            poe_range_end=360,
            elevation=np.empty((0, 0), dtype=np.float64),
            poe=np.empty((0, 0), dtype=np.float64),
            ir_er=np.empty((0, 0), dtype=np.float64),
            cumulative_motion=np.empty((0, 0), dtype=np.float64),
            sample_count=np.empty((0, 0), dtype=np.int32),
        )

    @pytest.fixture
    def postural_angles(self) -> MagicMock:
        """Create mock postural angles."""
        return MagicMock(spec=PosturalAngles)

    @pytest.fixture
    def relative_matrices(self):
        """Create mock relative rotation matrices."""
        return MagicMock()

    @pytest.fixture
    def bin_bounds(self) -> list[BinBounds]:
        """Create known bin bounds."""
        return [
            BinBounds(
                elevation_start=0,
                elevation_end=30,
                poe_start=0,
                poe_end=30,
            ),
            BinBounds(
                elevation_start=0,
                elevation_end=30,
                poe_start=30,
                poe_end=60,
            ),
            BinBounds(
                elevation_start=30,
                elevation_end=60,
                poe_start=0,
                poe_end=30,
            ),
        ]

    def test_calls_bin_functions_for_each_generated_bin(
        self,
        mocker,
        heatmap,
        relative_matrices,
        postural_angles,
        bin_bounds,
    ):
        """Calls bin calculation and addition once per bin."""

        result = MagicMock(spec=BinRotationResult)
        trace_totals = np.array([0.25, 0.5, 0.75], dtype=np.float64)
        euler_components = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float64,
        )

        generate_bins = mocker.patch(
            "modules.kinematics._generate_heatmap_bins",
            return_value=iter(bin_bounds),
        )
        calculate_trace = mocker.patch(
            "modules.kinematics._calculate_trace_rotation_angles",
            return_value=trace_totals,
        )
        decompose_euler = mocker.patch(
            "modules.kinematics._decompose_rotation_matrices_yxy",
            return_value=euler_components,
        )
        calculate_bin = mocker.patch(
            "modules.kinematics._calculate_single_bin",
            return_value=result,
        )
        add_result = mocker.patch(
            "modules.kinematics._add_bin_result_to_heatmap",
        )

        _ = _populate_heatmap(
            relative_matrices,
            postural_angles,
            heatmap,
        )

        calculate_trace.assert_called_once_with(relative_matrices)
        decompose_euler.assert_called_once_with(relative_matrices)
        generate_bins.assert_called_once_with(
            heatmap.bin_width,
            heatmap.elevation_range_end,
            heatmap.poe_range_end,
        )

        assert calculate_bin.call_count == len(bin_bounds)
        assert add_result.call_count == len(bin_bounds)

        for bounds in bin_bounds:
            calculate_bin.assert_any_call(
                traces=trace_totals,
                euler_components=euler_components,
                postural_angles=postural_angles,
                **bounds.__dict__,
            )

        for bounds in bin_bounds:
            add_result.assert_any_call(
                heatmap,
                result,
                bounds.elevation_start // heatmap.bin_width,
                bounds.poe_start // heatmap.bin_width,
            )

    def test_returns_heatmap_instance(
        self,
        mocker,
        heatmap,
        relative_matrices,
        postural_angles,
    ):
        mocker.patch(
            "modules.kinematics._generate_heatmap_bins",
            return_value=iter([]),
        )
        mocker.patch(
            "modules.kinematics._calculate_trace_rotation_angles",
            return_value=np.array([], dtype=np.float64),
        )
        mocker.patch(
            "modules.kinematics._decompose_rotation_matrices_yxy",
            return_value=np.empty((0, 3), dtype=np.float64),
        )

        result = _populate_heatmap(
            relative_matrices,
            postural_angles,
            heatmap,
        )

        assert isinstance(result, Heatmap)
        assert result is heatmap


class TestCalculateTraceRotationAngles:

    @pytest.mark.parametrize(
        "invalid_matrices",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64),
                            id="2x2-matrices"),
            pytest.param(np.zeros((3, 3, 4), dtype=np.float64),
                            id="wrong-last-dim"),
            pytest.param(np.zeros((4, 9), dtype=np.float64),
                            id="flattened-rows"),
            pytest.param(np.zeros((0, 3, 3), dtype=np.float64),
                            id="empty-matrices"),
        ],
    )
    def test_rejects_invalid_matrices_shape(self, invalid_matrices):
        with pytest.raises(ValueError):
            _calculate_trace_rotation_angles(invalid_matrices)

    def test_trace_angles_match_rotation_magnitude_formula(self):
        relative_matrices = np.array(
            [
                np.eye(3),
                R.from_euler("z", 90, degrees=True).as_matrix(),
                R.from_euler("x", 180, degrees=True).as_matrix(),
            ],
            dtype=np.float64,
        )

        result = _calculate_trace_rotation_angles(relative_matrices)

        expected = np.array(
            [0.0, np.pi / 2, np.pi],
            dtype=np.float64,
        )
        np.testing.assert_allclose(result, expected, atol=1e-8)

    def test_trace_angles_are_non_negative(self):
        relative_matrices = np.array(
            [
                R.from_euler("y", -15, degrees=True).as_matrix(),
                R.from_euler("x", -20, degrees=True).as_matrix(),
            ],
            dtype=np.float64,
        )

        result = _calculate_trace_rotation_angles(relative_matrices)

        assert np.all(result >= 0.0)

class TestCalculateBinRotations:
    @pytest.fixture
    def data(self) -> np.ndarray:
        """Create mock input rotation data."""
        return np.ones((10, 18))

    @pytest.fixture
    def validated_data(self) -> np.ndarray:
        """Create mock validated data."""
        return np.ones((10, 18), dtype=np.float64)

    @pytest.fixture
    def relative_matrices(self) -> np.ndarray:
        """Create mock relative rotation matrices."""
        return np.ones((9, 3, 3))

    @pytest.fixture
    def postural_angles(self) -> MagicMock:
        """Create mock postural angles."""
        return MagicMock()

    @pytest.fixture
    def heatmap(self) -> MagicMock:
        """Create mock heatmap instance."""
        return MagicMock(spec=Heatmap)

    def test_calls_dependencies_once_and_returns_heatmap(
        self,
        mocker,
        data,
        validated_data,
        relative_matrices,
        postural_angles,
        heatmap,
    ):
        """Calls each processing step once and returns a Heatmap."""

        mock_validate_matrices = mocker.patch(
            "modules.kinematics._validate_rotation_data",
            return_value=validated_data,
        )
        mock_create_relative_matrices_and_postures = mocker.patch(
            "modules.kinematics._create_relative_matrices_and_postural_angles",
            return_value=(
                relative_matrices,
                postural_angles,
            ),
        )
        mock_heatmap_constructor = mocker.patch(
            "modules.kinematics.Heatmap",
            return_value=heatmap,
        )
        mock_populate_heatmap = mocker.patch(
            "modules.kinematics._populate_heatmap",
            return_value=heatmap,
        )

        side="left"
        result = calculate_bin_rotations(
            data,
            side
        )

        mock_validate_matrices.assert_called_once_with(
            data,
            side
        )

        mock_create_relative_matrices_and_postures.assert_called_once_with(
            validated_data,
            side
        )

        mock_heatmap_constructor.assert_called_once_with()

        mock_populate_heatmap.assert_called_once_with(
            relative_matrices,
            postural_angles,
            heatmap,
        )

        assert isinstance(result, Heatmap)
        assert result is heatmap

