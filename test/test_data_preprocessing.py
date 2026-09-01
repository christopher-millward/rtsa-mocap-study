import pytest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as R

from modules.data_preprocessing import (
    apply_imu_calibration,
    clean_and_validate_data,
    get_correction_matrix,
    validate_orthonorm_and_det,
)

# ---- Global test variables ----
SMALL_ANGLE = 1e-3
SINGULARITY_TOLERANCE = 1e-7
TOLERANCE = 1e-9


# ---- Helper Functions ----
def _ensure_valid_R_matrices(matrices):
    # Check orthonormality: R.T @ R should be close to identity
    identity = np.eye(3)
    for i in range(matrices.shape[0]):
        if not np.allclose(matrices[i].T @ matrices[i], identity, atol=TOLERANCE):
            raise ValueError("matrices are not orthonormal")

        # Check determinant: should be close to 1
        if not np.isclose(np.linalg.det(matrices[i]), 1.0, atol=TOLERANCE):
            raise ValueError("matrices do not have determinant of 1")


# ---- Tests ----
class TestGetCorrectionMatrix:
    @pytest.mark.parametrize(
        "m, target",
        [
            pytest.param(np.zeros((0, 3), dtype=np.float64), np.eye(3), id="m-is-empty"),
            pytest.param(np.eye(3), np.zeros((0, 3), dtype=np.float64), id="target-is-empty"),
            pytest.param(np.zeros((2, 2), dtype=np.float64), np.eye(3), id="m-not-3x3"),
            pytest.param(np.eye(3), np.zeros((2, 2), dtype=np.float64), id="target-not-3x3"),
            pytest.param(np.zeros((3, 4), dtype=np.float64), np.eye(3), id="m-non-square"),
            pytest.param(np.eye(3), np.zeros((3, 4), dtype=np.float64), id="target-non-square"),
        ],
    )
    def test_should_reject_non_3x3_inputs(self, m, target):
        with pytest.raises(ValueError, match="m and target must have shape \\(3, 3\\)"):
            get_correction_matrix(m, target)

    @pytest.mark.parametrize(
        "m, target",
        [
            pytest.param(np.array([[1.0, "a", 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=object), np.eye(3), id="m-non-numeric"),
            pytest.param(np.eye(3), np.array([[1.0, "b", 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=object), id="target-non-numeric"),
            pytest.param(np.array([[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), np.eye(3), id="m-non-finite"),
            pytest.param(np.eye(3), np.array([[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64), id="target-non-finite"),
        ],
    )
    def test_should_reject_non_numeric_or_non_finite_inputs(self, m, target):
        with pytest.raises((TypeError, ValueError)):
            get_correction_matrix(m, target)

    def test_should_normalize_m_and_target_before_alignment(self):
        m = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
        target = np.array(
            [
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=np.float64,
        )

        mock_rot = patch("modules.data_preprocessing.Rotation.align_vectors").start()
        mock_rot.return_value = (R.from_euler("Z", 0.0), None)
        try:
            get_correction_matrix(m, target)
        finally:
            patch.stopall()

        mock_call = mock_rot.call_args[0]
        assert np.allclose(mock_call[0], m / np.linalg.norm(m))
        assert np.allclose(mock_call[1], target / np.linalg.norm(target))

    def test_should_call_align_vectors_with_m_and_target(self):
        m = np.eye(3, dtype=np.float64)
        target = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        expected = R.from_euler("Z", np.pi / 2.0).as_matrix()

        with patch("modules.data_preprocessing.Rotation.align_vectors", return_value=(R.from_matrix(expected), None)) as mock_align:
            result = get_correction_matrix(m, target)

        assert mock_align.call_count == 1
        assert result.shape == (3, 3)
        assert np.allclose(result, expected)

    @pytest.mark.parametrize(
        "m, target",
        [
            pytest.param(np.eye(3, dtype=np.float64), np.eye(3, dtype=np.float64), id="identity-to-identity"),
            pytest.param(R.from_euler("X", np.pi / 4.0).as_matrix(), np.eye(3, dtype=np.float64), id="x-rotation"),
            pytest.param(R.from_euler("Y", np.pi / 3.0).as_matrix(), np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64), id="arbitrary-rotation"),
        ],
    )
    def test_should_return_valid_rotation_matrix(self, m, target):
        result = get_correction_matrix(m, target)

        assert result.shape == (3, 3)
        assert np.allclose(result.T @ result, np.eye(3), atol=TOLERANCE)
        assert np.isclose(np.linalg.det(result), 1.0, atol=TOLERANCE)
        _ensure_valid_R_matrices(result.reshape(1, 3, 3))


class TestApplyAxisOrientationCorrection:

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.zeros((0, 0, 0), dtype=np.float64), id="empty-array"),
            pytest.param(np.zeros((10, 1, 3), dtype=np.float64), id="wrong-width"),
            pytest.param(np.zeros((10, 3, 3, 3), dtype=np.float64), id="4d-array"),
            pytest.param(np.zeros((5, 17), dtype=np.float64), id="wrong-flat-shape"),
        ],
    )
    def test_should_reject_invalid_shape(self, data):

        with pytest.raises(
            ValueError,
            match=r"data must have shape \(n_frames, 3, 3\)",
        ):
            apply_imu_calibration(data, n_frames=0)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.zeros((0, 3, 3), dtype=np.float64), id="empty-3x3-array"),
        ],
    )
    def test_should_reject_empty_data(self, data):
        with pytest.raises(
            ValueError,
            match="data must contain at least one frame",
        ):
            apply_imu_calibration(data, n_frames=0)

    def test_should_reject_n_frames_larger_than_data_length(self):
        data = np.tile(np.eye(3, dtype=np.float64), (3, 1, 1))
        with pytest.raises(ValueError, match="n_frames must not be greater than the number of frames in data"):
            apply_imu_calibration(data, n_frames=4)

    def test_should_calculate_average_humerus_direction_using_first_n_frames(self):
        data = np.stack(
            [
                R.from_euler("X", 0.1).as_matrix(),
                R.from_euler("Y", 0.2).as_matrix(),
                R.from_euler("Z", 0.3).as_matrix(),
            ],
            axis=0,
        )
        target = np.eye(3, dtype=np.float64)

        with patch(
            "modules.data_preprocessing.get_correction_matrix", 
            return_value=np.eye(3, dtype=np.float64)
        ) as mock_get_correction_matrix:
            apply_imu_calibration(data, n_frames=2, target=target)

        expected_avg = data[:2].mean(axis=0)
        mock_get_correction_matrix.assert_called_once()
        actual_avg, actual_target = mock_get_correction_matrix.call_args.args

        np.testing.assert_allclose(actual_avg, expected_avg)
        np.testing.assert_array_equal(actual_target, target)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.array([[[np.nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float64), id="nan-in-data"),
            pytest.param(np.array([[[np.inf, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float64), id="inf-in-data"),
        ],
    )
    def test_should_reject_non_finite_numeric_values(self, data):
        with pytest.raises(ValueError):
            apply_imu_calibration(data)

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=object), id="non-numeric-data"),
            pytest.param(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[2, 0, 0], [0, 2, 0], [0, 0, 2]]], dtype=object), id="object-array-data"),
        ],
    )
    def test_should_reject_non_numeric_values(self, data):
        with pytest.raises((TypeError, ValueError)):
            apply_imu_calibration(data)

    def test_should_apply_correction_matrix_to_every_frame(self):
        data = np.stack(
            [
                R.from_euler("X", 0.1).as_matrix(),
                R.from_euler("Y", 0.2).as_matrix(),
                R.from_euler("Z", 0.3).as_matrix(),
            ],
            axis=0,
        )
        correction = R.from_euler("Z", np.pi / 2.0).as_matrix()

        with patch("modules.data_preprocessing.get_correction_matrix", return_value=correction):
            result = apply_imu_calibration(data, n_frames=data.shape[0])

        expected = correction @ data
        assert np.allclose(result, expected)

    def test_should_return_valid_rotation_matrices_for_valid_input(self):
        data = np.stack(
            [
                R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix(),
                R.from_euler("XYZ", [0.4, -0.2, 0.5]).as_matrix(),
                R.from_euler("XYZ", [-0.3, 0.6, 0.1]).as_matrix(),
            ],
            axis=0,
        )

        result = apply_imu_calibration(data, n_frames=2)

        assert result.shape == data.shape
        _ensure_valid_R_matrices(result)

    @pytest.mark.parametrize("angle", [0.0, 1e-9, 1e-6, 1e-3])
    def test_should_remain_stable_for_rotations_close_to_identity(self, angle):
        data = np.tile(R.from_euler("X", angle).as_matrix(), (5, 1, 1))

        result = apply_imu_calibration(data, n_frames=data.shape[0])

        assert result.shape == data.shape
        _ensure_valid_R_matrices(result)

    @pytest.mark.parametrize(
        "angle",
        [
            0.0 + SINGULARITY_TOLERANCE,
            np.pi - SINGULARITY_TOLERANCE,
        ],
    )
    def test_should_handle_matrices_near_singularities(self, angle):
        data = np.tile(R.from_euler("Y", angle).as_matrix(), (5, 1, 1))

        result = apply_imu_calibration(data, n_frames=data.shape[0])

        assert result.shape == data.shape
        _ensure_valid_R_matrices(result)


class TestValidateOrthonormAndDet:
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

    def test_should_reject_empty_batch(self):
        matrices = np.zeros((0, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="batch must contain at least one matrix"):
            validate_orthonorm_and_det(matrices)

    def test_should_perform_calculations_using_float64_precision(self):
        data = R.from_euler("X", 0.1).as_matrix().reshape(1, 3, 3)

        with patch("numpy.asarray", wraps=np.asarray) as spy_asarray:
            validate_orthonorm_and_det(data)

        assert any(
            call.args[0] is data and call.kwargs.get("dtype") == np.float64
            for call in spy_asarray.call_args_list
        )

    @pytest.mark.parametrize(
        "matrix",
        [
            pytest.param(R.from_euler("X", np.pi / 5).as_matrix().reshape(1, 3, 3), id="x-rotation"),
            pytest.param(R.from_euler("Y", np.pi / 5).as_matrix().reshape(1, 3, 3), id="y-rotation"),
            pytest.param(R.from_euler("Z", np.pi / 5).as_matrix().reshape(1, 3, 3), id="z-rotation"),
            pytest.param(R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix().reshape(1, 3, 3), id="xyz-rotation"),
            pytest.param(R.from_euler("X", SMALL_ANGLE).as_matrix().reshape(1, 3, 3), id="small-angle-rotation"),
            pytest.param(R.from_euler("Y", np.pi / 2).as_matrix().reshape(1, 3, 3), id="90-degree-rotation"),
        ],
    )
    def test_should_accept_valid_rotation_matrices(self, matrix):
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for valid rotation matrices.")

    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.tile(R.from_euler("X", 0.1).as_matrix(), (5, 1, 1)), id="batch-of-x-rotations"),
            pytest.param(np.tile(R.from_euler("Y", 0.1).as_matrix(), (5, 1, 1)), id="batch-of-y-rotations"),
            pytest.param(np.tile(R.from_euler("Z", 0.1).as_matrix(), (5, 1, 1)), id="batch-of-z-rotations"),
            pytest.param(np.tile(R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix(), (5, 1, 1)), id="batch-of-xyz-rotations"),
        ],
    )
    def test_should_handle_batches_of_valid_rotation_matrices(self, matrices):
        try:
            validate_orthonorm_and_det(matrices)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for a batch of valid rotation matrices.")

    def test_should_reject_non_orthonormal_matrices(self):
        matrices = np.ones((1, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError, match="matrices must be orthonormal rotation matrices"):
            validate_orthonorm_and_det(matrices)

    def test_should_reject_non_one_determinant(self):
        matrices = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        matrices[0][0][0] = -1
        with pytest.raises(ValueError, match="matrices must have a determinant of 1"):
            validate_orthonorm_and_det(matrices)

    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.ones((1, 3, 3), dtype=np.float64), id="non-orthonormal"),
            pytest.param(np.zeros((1, 3, 3), dtype=np.float64), id="zero-matrix"),
            pytest.param(np.concatenate((np.tile(R.from_euler("X", 0.1).as_matrix(), (4, 1, 1)), np.ones((1, 3, 3), dtype=np.float64)), axis=0), id="batch-with-one-invalid"),
        ],
    )
    def test_should_reject_batches_with_any_invalid_matrices(self, matrices):
        with pytest.raises(ValueError):
            validate_orthonorm_and_det(matrices)

    @pytest.mark.parametrize(
        "angle",
        [
            0.0 + SINGULARITY_TOLERANCE,
            np.pi - SINGULARITY_TOLERANCE,
        ],
    )
    def test_should_handle_matrices_near_singularities(self, angle):
        matrix = R.from_euler("Y", angle).as_matrix().reshape(1, 3, 3)
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail("validate_orthonorm_and_det raised ValueError unexpectedly for a valid rotation matrix near a singularity.")


class TestCleanAndValidateData:
    """Tests for clean_and_validate_data."""

    def test_should_coerce_input_to_float64(self):
        """Should pass float64 data to the axis correction function."""
        raw_data = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32,
        )

        with patch(
            "modules.data_preprocessing.apply_imu_calibration",
            return_value=raw_data.astype(np.float64),
        ) as mock_apply:
            clean_and_validate_data(raw_data) #type:ignore

        mock_apply.assert_called_once()
        actual_data = mock_apply.call_args.args[0]
        assert actual_data.dtype == np.float64
        np.testing.assert_array_equal(actual_data, raw_data.astype(np.float64))


    def test_should_call_apply_imu_calibration_with_coerced_data(
        self
    ):
        """Should pass coerced float64 data to imu calibration."""
        raw_data = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        expected_data = raw_data.astype(np.float64)
        corrected_data = np.ones((1, 3, 3), dtype=np.float64)

        with patch(
            "modules.data_preprocessing.apply_imu_calibration",
            return_value=corrected_data,
        ) as mock_apply, patch(
            "modules.data_preprocessing.validate_orthonorm_and_det",
        ):
            clean_and_validate_data(raw_data)#type:ignore

        mock_apply.assert_called_once()

        actual_data = mock_apply.call_args.args[0]

        np.testing.assert_array_equal(actual_data, expected_data)

    def test_should_call_validate_with_cleaned_data(self):
        """Should validate the data returned by axis correction."""
        raw_data = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        cleaned_data = np.array(
            [[[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=np.float64,
        )

        with patch(
            "modules.data_preprocessing.apply_imu_calibration",
            return_value=cleaned_data,
        ), patch(
            "modules.data_preprocessing.validate_orthonorm_and_det",
        ) as mock_validate:
            clean_and_validate_data(raw_data)#type:ignore

        mock_validate.assert_called_once()
        actual_data = mock_validate.call_args.args[0]
        np.testing.assert_array_equal(actual_data, cleaned_data)

    def test_should_return_cleaned_data(self):
        """Should return the data produced by imu calibration."""
        raw_data = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
            dtype=np.float32,
        )
        cleaned_data = np.array(
            [[[0, 1, 0], [1, 0, 0], [0, 0, 1]]],
            dtype=np.float64,
        )

        with patch(
            "modules.data_preprocessing.apply_imu_calibration",
            return_value=cleaned_data,
        ), patch(
            "modules.data_preprocessing.validate_orthonorm_and_det",
        ):
            result = clean_and_validate_data(raw_data) #type:ignore

        np.testing.assert_array_equal(result, cleaned_data)