import pytest
from unittest.mock import patch
from scipy.spatial.transform import Rotation as R
import numpy as np
from numpy import typing as npt
from modules.data_preprocessing import validate_orthonorm_and_det, align_axes_with_ISB

# ---- Global test variables ----
SMALL_ANGLE = 1e-3
SINGULARITY_TOLERANCE = 1e-7
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
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

# ---- Helper Functions ----
def _flatten_arm_matrices(
    left: npt.NDArray[np.float64],
    right: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Combine left and right rotation matrices into an (N, 18) array."""

    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape")

    return np.concatenate(
        [
            left.reshape(-1, 9),
            right.reshape(-1, 9),
        ],
        axis=1,
    )

def _unflatten_arm_matrices(
    raw_data: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split an (N, 18) array into left and right rotation matrices."""

    if raw_data.ndim != 2 or raw_data.shape[1] != 18:
        raise ValueError(
            f"Expected raw_data shape (N,18), got {raw_data.shape}"
        )

    return (
        raw_data[:, :9].reshape(raw_data.shape[0], 3, 3),
        raw_data[:, 9:18].reshape(raw_data.shape[0], 3, 3),
    )

def _expected_alignment(
    raw_data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the expected ISB alignment using matrix multiplication.

    This helper is intended for unit testing. It constructs full 3x3 rotation
    matrices for each arm, applies the change-of-basis transformation using
    matrix multiplication, and returns the transformed matrices in the same
    flattened (N, 18) format expected by ``align_axes_with_ISB_flat``.
    """
    left, right = _unflatten_arm_matrices(raw_data)

    left = TORSO_TO_ISB @ left @ LEFT_TO_ISB.T
    right = TORSO_TO_ISB @ right @ RIGHT_TO_ISB.T

    return _flatten_arm_matrices(left, right)

# ---- Tests ----
class TestAlignAxesWithISB:

    # Should reject non-x18 inputs.
    @pytest.mark.parametrize(
        "raw_data",
        [
            np.zeros(18),
            np.zeros((18,)),
            np.zeros((5, 17)),
            np.zeros((5, 19)),
            np.zeros((5, 18, 1)),
        ],
    )
    def test_should_reject_invalid_shapes(self, raw_data):
        with pytest.raises(ValueError, match=r"raw_data must have shape \(n_frames, 18\)"):
            align_axes_with_ISB(raw_data)

    # Should reject empty batches.
    def test_should_reject_empty_batch(self):
        raw_data = np.zeros((0, 18), dtype=np.float64)
        with pytest.raises(ValueError, match="raw_data must contain at least one frame"):
            align_axes_with_ISB(raw_data)

    # Should reject non-numeric values during float conversion.
    @pytest.mark.parametrize(
        "raw_data",
        [
            pytest.param(np.array([[None if i == 5 else i for i in range(18)]], dtype=object), id="none"),
            pytest.param(np.array([["x" if i == 5 else i for i in range(18)]], dtype=object), id="string"),
        ],
    )
    def test_should_reject_non_numeric_values(self, raw_data):
        with pytest.raises(ValueError, match=r"raw_data must contain only (numeric values|finite values)"):
            align_axes_with_ISB(raw_data)

    # Should reject non-finite numeric values.
    @pytest.mark.parametrize(
        "raw_data",
        [
            pytest.param(np.array([[np.nan if i == 5 else i for i in range(18)]], dtype=object), id="nan"),
            pytest.param(np.array([[np.inf if i == 5 else i for i in range(18)]], dtype=object), id="inf"),
        ],
    )
    def test_should_reject_non_finite_values(self, raw_data):
        with pytest.raises(ValueError, match="raw_data must contain only finite values"):
            align_axes_with_ISB(raw_data)

    # Should align the identity matrix correctly.
    def test_should_align_identity_rotation(self):
        left = np.eye(3).reshape(1,3,3)
        right = np.eye(3).reshape(1,3,3)

        raw = _flatten_arm_matrices(left, right)
        expected = _expected_alignment(raw)
        result = align_axes_with_ISB(raw)

        assert np.allclose(result, expected)

    # Should align 90-degree rotations about all axes.
    @pytest.mark.parametrize("axis", ["X","Y","Z"])
    def test_should_align_90_degree_rotations(self, axis):
        left = R.from_euler(axis, np.pi/2).as_matrix().reshape(1,3,3)
        right = R.from_euler(axis, -np.pi/2).as_matrix().reshape(1,3,3)

        raw_data = _flatten_arm_matrices(left, right)
        expected = _expected_alignment(raw_data)
        result = align_axes_with_ISB(raw_data)

        assert np.allclose(result, expected)

    # Should align batches of matrices frame by frame.
    def test_should_align_multiple_matrices(self):
        raw_data = np.concatenate(
            [
                _flatten_arm_matrices(R.from_euler("X", 0.1).as_matrix(), np.zeros((3, 3))),
                _flatten_arm_matrices(R.from_euler("Y", 0.2).as_matrix(), np.zeros((3, 3))),
                _flatten_arm_matrices(R.from_euler("Z", 0.3).as_matrix(), np.zeros((3, 3))),
                _flatten_arm_matrices(R.from_euler("XYZ", [0.1, 0.2, 0.3]).as_matrix(), np.zeros((3, 3))),
            ],
            axis=0,
        )
        expected = _expected_alignment(raw_data)
        result = align_axes_with_ISB(raw_data)

        assert np.allclose(result, expected)

    # Should preserve orthonormality for valid rotation matrices.
    def test_should_preserve_orthonormality(self):
        raw_left = R.from_euler("X", np.pi/2).as_matrix().reshape(1,3,3)
        raw_right = R.from_euler("X", -np.pi/2).as_matrix().reshape(1,3,3)
        raw_data = _flatten_arm_matrices(raw_left, raw_right)

        result = align_axes_with_ISB(raw_data)
        result_left = result[:, :9].reshape(-1,3,3)
        result_right = result[:, 9:].reshape(-1,3,3)

        for matrices in (result_left, result_right):
            gram = matrices.transpose(0,2,1) @ matrices
            identity = np.broadcast_to(np.eye(3), gram.shape)

            assert np.allclose(gram, identity, atol=TOLERANCE)

    # Should preserve a determinant of +1 for proper rotation matrices.
    def test_should_preserve_determinant(self):
        raw_left = R.from_euler("X", np.pi/2).as_matrix().reshape(1,3,3)
        raw_right = R.from_euler("X", -np.pi/2).as_matrix().reshape(1,3,3)
        raw_data = _flatten_arm_matrices(raw_left, raw_right)

        result = align_axes_with_ISB(raw_data)
        result_left = result[:, :9].reshape(-1,3,3)
        result_right = result[:, 9:].reshape(-1,3,3)

        assert np.allclose(np.linalg.det(result_left), 1.0, atol=TOLERANCE)
        assert np.allclose(np.linalg.det(result_right), 1.0, atol=TOLERANCE)

    # Should recover the original matrices when the inverse transform is applied.
    def test_should_recover_original_matrix_with_inverse_transform(self):
        raw_left = R.from_euler("X", np.pi/2).as_matrix().reshape(1,3,3)
        raw_right = R.from_euler("X", -np.pi/2).as_matrix().reshape(1,3,3)
        raw_data = _flatten_arm_matrices(raw_left, raw_right)

        result = align_axes_with_ISB(raw_data)
        result_left = result[:, :9].reshape(-1,3,3)
        result_right = result[:, 9:].reshape(-1,3,3)

        left_recovered = (
            TORSO_TO_ISB.T
            @ result_left
            @ LEFT_TO_ISB
        )

        right_recovered = (
            TORSO_TO_ISB.T
            @ result_right
            @ RIGHT_TO_ISB
        )

        assert np.allclose(left_recovered, raw_left, atol=TOLERANCE)
        assert np.allclose(right_recovered, raw_right, atol=TOLERANCE)

    # Should remain stable for rotations very close to identity.
    def test_should_handle_rotations_close_to_identity(self):
        left = R.from_euler("Y", 1e-10).as_matrix().reshape(1, 3, 3)
        right = R.from_euler("Y", 1e-10).as_matrix().reshape(1, 3, 3)
        raw_data = _flatten_arm_matrices(left, right)
        expected = _expected_alignment(raw_data)

        result = align_axes_with_ISB(raw_data)

        assert np.allclose(result, expected, atol=TOLERANCE)

    # Should produce a different result when the transformation is applied twice.
    def test_should_not_match_single_application_after_repeated_application(self):
        left = R.from_euler( "XYZ", [0.25, 0.5, 0.75]).as_matrix().reshape(1, 3, 3)
        right = R.from_euler( "XYZ", [0.25, 0.5, 0.75]).as_matrix().reshape(1, 3, 3)
        raw_data = _flatten_arm_matrices(left, right)
        single_application = align_axes_with_ISB(raw_data)
        repeated_application = align_axes_with_ISB(single_application)

        assert not np.allclose(repeated_application, single_application)


class TestValidateOrthonormAndDet:
    # Should reject non-3x3 matrices
    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64),
                         id="2x2-matrices"),
            pytest.param(np.zeros((2, 3, 4), dtype=np.float64),
                         id="non-square-matrices"),
            pytest.param(np.zeros((2, 3), dtype=np.float64),
                         id="flat-matrices"),
            pytest.param(np.zeros((2, 4, 4), dtype=np.float64),
                         id="4x4-matrices"),
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
            pytest.param(R.from_euler(
                'X', np.pi / 5).as_matrix().reshape(1, 3, 3), id="x-rotation"),
            pytest.param(R.from_euler(
                'Y', np.pi / 5).as_matrix().reshape(1, 3, 3), id="y-rotation"),
            pytest.param(R.from_euler(
                'Z', np.pi / 5).as_matrix().reshape(1, 3, 3), id="z-rotation"),
            pytest.param(R.from_euler('XYZ', [0.1, 0.2, 0.3]).as_matrix().reshape(
                1, 3, 3), id="xyz-rotation"),
            pytest.param(R.from_euler('X', SMALL_ANGLE).as_matrix().reshape(
                1, 3, 3), id="small-angle-rotation"),
            pytest.param(R.from_euler(
                'Y', np.pi / 2).as_matrix().reshape(1, 3, 3), id="90-degree-rotation"),
        ]
    )
    def test_should_accept_valid_rotation_matrices(self, matrix):
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail(
                "validate_orthonorm_and_det raised ValueError unexpectedly for valid rotation matrices.")

    # Should handle batches of valid rotation matrices
    @pytest.mark.parametrize(
        "matrices",
        [
            pytest.param(np.tile(R.from_euler('X', 0.1).as_matrix(),
                         (5, 1, 1)), id="batch-of-x-rotations"),
            pytest.param(np.tile(R.from_euler('Y', 0.1).as_matrix(),
                         (5, 1, 1)), id="batch-of-y-rotations"),
            pytest.param(np.tile(R.from_euler('Z', 0.1).as_matrix(),
                         (5, 1, 1)), id="batch-of-z-rotations"),
            pytest.param(np.tile(R.from_euler('XYZ', [0.1, 0.2, 0.3]).as_matrix(
            ), (5, 1, 1)), id="batch-of-xyz-rotations"),
        ]
    )
    def test_should_handle_batches_of_valid_rotation_matrices(self, matrices):
        try:
            validate_orthonorm_and_det(matrices)
        except ValueError:
            pytest.fail(
                "validate_orthonorm_and_det raised ValueError unexpectedly for a batch of valid rotation matrices.")

    # Should reject non-orthonormal matrices
    def test_should_reject_non_orthonormal_matrices(self):
        matrices = np.ones((1, 3, 3), dtype=np.float64)  # not orthonormal
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
            pytest.param(np.ones((1, 3, 3), dtype=np.float64),
                         id="non-orthonormal"),
            pytest.param(np.zeros((1, 3, 3), dtype=np.float64),
                         id="zero-matrix"),
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
            0.0 + SINGULARITY_TOLERANCE,
            np.pi - SINGULARITY_TOLERANCE
        ]
    )
    def test_should_handle_matrices_near_singularities(self, angle):
        matrix = R.from_euler('Y', angle).as_matrix().reshape(1, 3, 3)
        try:
            validate_orthonorm_and_det(matrix)
        except ValueError:
            pytest.fail(
                "validate_orthonorm_and_det raised ValueError unexpectedly for a valid rotation matrix near a singularity.")
