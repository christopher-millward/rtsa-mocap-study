import numpy as np
from scipy.spatial.transform import Rotation as R
import pytest
import numpy.typing as npt

# Use local trace formula in tests; don't import production helper here.
from utils.kinematics.individual_axes import (
    compute_incremental_rotation_matrices,
    decompose_rotation_matrices_yxy,
)

# ---- Global test variables ----
small_angle = 1e-3
tolerance = 1e-9
singularity_tolerance = 1e-7


# ---- Helper Functions ----
def _is_rotation_matrix(R) -> bool:
    """Sanity check orthonormality."""
    return np.allclose(R.T @ R, np.eye(3), atol=tolerance)


class TestComputeIncrementalRotationMatrices:
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64), id="2x2-matrices"),
            pytest.param(np.zeros((3, 3, 4), dtype=np.float64), id="wrong-last-dim"),
            pytest.param(np.zeros((4, 9), dtype=np.float64), id="flattened-rows"),
        ],
    )
    def test_should_reject_non_3x3_inputs(self, rotation_matrices):
        """The function should only accept batches of 3x3 matrices."""
        with pytest.raises(ValueError):
            compute_incremental_rotation_matrices(rotation_matrices)

    # Test that fewer than two frames are rejected with a ValueError.
    @pytest.mark.parametrize(
        "rotation_matrices",
        [
            pytest.param(np.zeros((0, 3, 3), dtype=np.float64), id="zero-frames"),
            pytest.param(np.stack([np.eye(3, dtype=np.float64)]), id="single-frame"),
        ],
    )
    def test_should_reject_insufficient_frames(self, rotation_matrices):
        """The function should require at least two frames so a relative rotation exists."""
        with pytest.raises(ValueError):
            compute_incremental_rotation_matrices(rotation_matrices)

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
        deltas = compute_incremental_rotation_matrices(rotation_matrices)
        assert deltas.dtype == np.float64

    # Test that a sequence of identical matrices produces identity deltas.
    @pytest.mark.parametrize("n_frames", [2, 5, 10])
    def test_should_return_identity_deltas_for_constant_sequence(self, n_frames):
        """A constant absolute-orientation sequence should produce identity deltas."""
        frames = np.stack([np.eye(3, dtype=np.float64) for _ in range(n_frames)])
        deltas = compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (n_frames - 1, 3, 3)
        for D in deltas:
            assert np.allclose(D, np.eye(3), atol=tolerance)

    # Test that the function correctly computes the relative rotation.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X", np.pi / 6, id="x-axis"),
            pytest.param(R.from_euler, "Y", np.pi / 4, id="y-axis"),
            pytest.param(R.from_euler, "Z", np.pi / 3, id="z-axis"),
            pytest.param(R.from_euler, "XYZ", [0.2, 0.2, 0.2], id="combined-xyz"),
        ],
    )
    def test_should_return_expected_relative_matrix(self, rotation_builder, sequence, angle):
        """For a two-frame sequence, the returned delta should equal R_current @ R_previous.T."""
        R0 = np.eye(3, dtype=np.float64)
        R1 = rotation_builder(sequence, angle).as_matrix()
        frames = np.stack([R0, R1])
        deltas = compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (1, 3, 3)
        expected = R1 @ R0.T
        assert np.allclose(deltas[0], expected, atol=tolerance)

    # Test that the cumulative product of deltas reconstructs the original sequence.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence","angle", "n_steps"),
        [
            pytest.param(R.from_euler, "X", np.pi/12, 4, id="small-x"),
            pytest.param(R.from_euler, "Y", np.pi/12, 4, id="small-y"),
            pytest.param(R.from_euler, "Z", np.pi/12, 4, id="small-z"),
            pytest.param(R.from_euler, "YXY", [0.15, 0.15, 0.15], 4, id="small-yxy"),
        ],
    )
    def test_should_reconstruct_to_absolute_orientation(self, rotation_builder, sequence, angle, n_steps):
        """The cumulative product of deltas should reproduce the original sequence."""
        D = rotation_builder(sequence, angle).as_matrix()
        frames = [np.eye(3, dtype=np.float64)]
        for _ in range(n_steps - 1):
            frames.append(frames[-1] @ D)
        frames = np.stack(frames)

        deltas = compute_incremental_rotation_matrices(frames)

        # Reconstruct sequential frames from R0 and deltas
        reconstructed = [frames[0]]
        for i in range(deltas.shape[0]):
            reconstructed.append(reconstructed[-1] @ deltas[i])
        reconstructed = np.stack(reconstructed)

        assert np.allclose(reconstructed, frames, atol=tolerance)

    # Test that each delta is a valid rotation matrix (orthonormal with determinant 1).
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X",  np.pi / 8, id="x-axis"),
            pytest.param(R.from_euler, "Y", np.pi/ 7, id="y-axis"),
            pytest.param(R.from_euler, "Z", np.pi/ 9, id="z-axis"),
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

        deltas = compute_incremental_rotation_matrices(frames)
        for D in deltas:
            assert _is_rotation_matrix(D)

    # Test that the function handles small and large rotations appropriately.
    @pytest.mark.parametrize(
        ("rotation_builder", "sequence", "angle"),
        [
            pytest.param(R.from_euler, "X",  small_angle, id="small-x"),
            pytest.param(R.from_euler, "Y", small_angle, id="small-y"),
            pytest.param(R.from_euler, "Z", small_angle, id="small-z"),
            pytest.param(R.from_euler, "YXY", [small_angle, small_angle, small_angle], id="small-yxy"),
            pytest.param(R.from_euler, "X",  np.pi / 2, id="large-x"),
            pytest.param(R.from_euler, "Y", np.pi * 0.75, id="large-y"),
            pytest.param(R.from_euler, "Z", np.pi * 0.9, id="large-z"),
            pytest.param(R.from_euler, "YXY", [np.pi / 6, np.pi / 6, np.pi / 6], id="large-yxy"),
        ]
    )
    def test_should_handle_small_and_large_rotations(self, rotation_builder, sequence, angle):
        """Small absolute changes should yield small deltas, and larger changes should yield larger deltas."""
        R0 = np.eye(3, dtype=np.float64)
        R1 = rotation_builder(sequence, angle).as_matrix()
        deltas = compute_incremental_rotation_matrices(np.stack([R0, R1]))
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

        deltas = compute_incremental_rotation_matrices(frames)
        assert deltas.shape == (n_frames - 1, 3, 3)
        assert deltas.dtype == np.float64

class TestDecomposeRotationMatricesYXY:
    # Test that non-3x3 inputs are rejected with a ValueError.
    @pytest.mark.parametrize(
        "relative_rotations",
        [
            pytest.param(np.zeros((2, 2, 2), dtype=np.float64), id="2x2-matrices"),
            pytest.param(np.zeros((3, 3, 4), dtype=np.float64), id="wrong-last-dim"),
            pytest.param(np.zeros((4, 9), dtype=np.float64), id="flattened-rows"),
        ],
    )
    def test_should_reject_non_3x3_inputs(self, relative_rotations):
        with pytest.raises(ValueError):
            decompose_rotation_matrices_yxy(relative_rotations)

    # Test that empty batch input raises a ValueError.
    @pytest.mark.parametrize(
        "relative_rotations",
        [
            pytest.param(np.stack([np.eye(3, dtype=np.float32)]), id="float32"),
            pytest.param(np.stack([np.eye(3, dtype=np.int64)]), id="int64"),
        ],
    )
    def test_should_coerce_non_float64_inputs(self, relative_rotations):
        angles = decompose_rotation_matrices_yxy(relative_rotations)
        assert angles.dtype == np.float64

    # Test that empty batch input raises a ValueError.
    def test_should_reject_empty_batch(self):
        empty = np.zeros((0, 3, 3), dtype=np.float64)
        with pytest.raises(ValueError):
            decompose_rotation_matrices_yxy(empty)

    # Output shape and dtype for single and multiple steps
    @pytest.mark.parametrize("n_steps", [1, 5])
    def test_should_return_expected_shape_and_dtype(self, n_steps):
        matrices = np.stack([
            R.from_euler("YXY", [0.1, 0.05, 0.2]).as_matrix()
            for _ in range(n_steps)
        ])
        angles = decompose_rotation_matrices_yxy(matrices)
        assert angles.shape == (n_steps, 3)
        assert angles.dtype == np.float64

    # Identity matrices should yield zero angles
    def test_identity_decomposes_to_zeros(self):
        matrices = np.stack([np.eye(3, dtype=np.float64) for _ in range(3)])
        angles = decompose_rotation_matrices_yxy(matrices)
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
        angles = decompose_rotation_matrices_yxy(matrices)[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, matrices[0], atol=tolerance)

    # Singularity: middle angle (X) near zero should still reconstruct
    @pytest.mark.parametrize("beta", [0.0, 1e-8])
    def test_singularity_beta_near_zero(self, beta):
        a, c = 0.3, -0.4
        M = R.from_euler("YXY", [a, beta, c]).as_matrix()
        angles = decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=singularity_tolerance)

    # Singularity: middle angle (X) near pi should still reconstruct
    def test_singularity_beta_near_pi(self):
        beta = np.pi - 1e-8
        a, c = 0.2, 0.5
        M = R.from_euler("YXY", [a, beta, c]).as_matrix()
        angles = decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=singularity_tolerance)

    # Clipping robustness: tiny noise pushing values slightly outside [-1,1]
    def test_clipping_robustness(self):
        a, b, c = 0.4, 0.9, -0.2
        M = R.from_euler("YXY", [a, b, c]).as_matrix()
        noisy = M.copy()
        noisy += np.random.default_rng(1).normal(scale=1e-12, size=M.shape)
        angles = decompose_rotation_matrices_yxy(np.stack([noisy]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=tolerance)

    # Non-rotation matrices should raise ValueError
    def test_should_reject_non_rotation_matrices(self):
        bad = np.eye(3, dtype=np.float64)
        bad[0] *= 2.0  # break orthonormality
        with pytest.raises(ValueError):
            decompose_rotation_matrices_yxy(np.stack([bad]))

    # Determinism: repeated calls return identical results
    def test_deterministic_outputs(self):
        M = R.from_euler("YXY", [0.25, 0.15, -0.35]).as_matrix()  # Use the same matrix for consistency
        first = decompose_rotation_matrices_yxy(np.stack([M]))
        second = decompose_rotation_matrices_yxy(np.stack([M]))
        assert np.allclose(first, second)

    # Sensitivity across magnitudes: tiny and near-pi angles reconstruct
    @pytest.mark.parametrize(
        ("a", "b", "c"),
        [
            pytest.param(small_angle, small_angle, -small_angle, id="tiny-angles"),
            pytest.param(1.2, np.pi - small_angle, -0.9, id="large-middle"),
        ],
    )
    def test_small_and_large_angle_sensitivity(self, a, b, c):
        M = R.from_euler("YXY", [a, b, c]).as_matrix()
        angles = decompose_rotation_matrices_yxy(np.stack([M]))[0]
        recomposed = R.from_euler("YXY", angles).as_matrix()
        assert np.allclose(recomposed, M, atol=tolerance)
