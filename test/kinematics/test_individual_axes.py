import numpy as np
import pytest

# Use local trace formula in tests; don't import production helper here.
from utils.kinematics.individual_axes import (
    compute_incremental_rotation_matrices,
)

# ---- Helper Functions ----
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


# --- Tests for compute_incremental_rotation_matrices ---
# Test that non-3x3 inputs are rejected with a ValueError.
@pytest.mark.parametrize(
    "rotation_matrices",
    [
        pytest.param(np.zeros((2, 2, 2), dtype=np.float64), id="2x2-matrices"),
        pytest.param(np.zeros((3, 3, 4), dtype=np.float64), id="wrong-last-dim"),
        pytest.param(np.zeros((4, 9), dtype=np.float64), id="flattened-rows"),
    ],
)
def test_compute_incremental_rotation_matrices_rejects_non_3x3_inputs(rotation_matrices):
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
def test_compute_incremental_rotation_matrices_rejects_insufficient_frames(rotation_matrices):
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
def test_compute_incremental_rotation_matrices_coerces_non_float64_inputs(rotation_matrices):
    """The function should coerce input matrices to float64."""
    deltas = compute_incremental_rotation_matrices(rotation_matrices)
    assert deltas.dtype == np.float64

# Test that a sequence of identical matrices produces identity deltas.
@pytest.mark.parametrize("n_frames", [2, 5, 10])
def test_compute_incremental_rotation_matrices_identity_sequence_returns_identity_deltas(n_frames):
    """A constant absolute-orientation sequence should produce identity deltas."""
    frames = np.stack([np.eye(3, dtype=np.float64) for _ in range(n_frames)])
    deltas = compute_incremental_rotation_matrices(frames)
    assert deltas.shape == (n_frames - 1, 3, 3)
    for D in deltas:
        assert np.allclose(D, np.eye(3), atol=1e-12)

# Test that the function correctly computes the relative rotation.
@pytest.mark.parametrize(
    ("rotation_builder", "angle"),
    [
        pytest.param(_rotation_matrix_x, np.pi / 6, id="x-axis"),
        pytest.param(_rotation_matrix_y, np.pi / 4, id="y-axis"),
        pytest.param(_rotation_matrix_z, np.pi / 3, id="z-axis"),
        pytest.param(lambda a: _rotation_matrix_xyz(a, a, a), 0.2, id="combined-xyz"),
    ],
)
def test_compute_incremental_rotation_matrices_returns_expected_relative_matrix(rotation_builder, angle):
    """For a two-frame sequence, the returned delta should equal R_current @ R_previous.T."""
    R0 = np.eye(3, dtype=np.float64)
    R1 = rotation_builder(angle)
    frames = np.stack([R0, R1])
    deltas = compute_incremental_rotation_matrices(frames)
    assert deltas.shape == (1, 3, 3)
    expected = R1 @ R0.T
    assert np.allclose(deltas[0], expected, atol=1e-12)

# Test that the cumulative product of deltas reconstructs the original sequence.
@pytest.mark.parametrize(
    ("rotation_builder", "angle", "n_steps"),
    [
        pytest.param(_rotation_matrix_x, np.pi / 12, 4, id="small-x"),
        pytest.param(_rotation_matrix_y, np.pi / 8, 6, id="small-y"),
        pytest.param(_rotation_matrix_z, np.pi / 10, 5, id="small-z"),
        pytest.param(lambda a: _rotation_matrix_xyz(a, a, a), 0.15, 5, id="small-xyz"),
    ],
)
def test_compute_incremental_rotation_matrices_reconstructs_absolute_orientation(rotation_builder, angle, n_steps):
    """The cumulative product of deltas should reproduce the original sequence."""
    D = rotation_builder(angle)
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

    assert np.allclose(reconstructed, frames, atol=1e-12)

# Test that each delta is a valid rotation matrix (orthonormal with determinant 1).
@pytest.mark.parametrize(
    ("rotation_builder", "angle"),
    [
        pytest.param(_rotation_matrix_x, np.pi / 8, id="x-axis"),
        pytest.param(_rotation_matrix_y, np.pi / 7, id="y-axis"),
        pytest.param(_rotation_matrix_z, np.pi / 9, id="z-axis"),
        pytest.param(lambda a: _rotation_matrix_xyz(a, 2 * a, 3 * a), 0.1, id="combined"),
    ],
)
def test_compute_incremental_rotation_matrices_returns_rotation_matrices(rotation_builder, angle):
    """Each relative matrix should still be a valid rotation matrix."""
    n_steps = 5
    D = rotation_builder(angle)
    frames = [np.eye(3, dtype=np.float64)]
    
    for _ in range(n_steps - 1):
        frames.append(frames[-1] @ D)
    frames = np.stack(frames)

    deltas = compute_incremental_rotation_matrices(frames)
    for D in deltas:
        assert _is_rotation_matrix(D)

# Test that the function handles small and large rotations appropriately.
@pytest.mark.parametrize(
    ("rotation_builder", "angle"),
    [
        pytest.param(_rotation_matrix_x, 1e-3, id="small-x"),
        pytest.param(_rotation_matrix_y, 2e-3, id="small-y"),
        pytest.param(_rotation_matrix_z, 5e-3, id="small-z"),
        pytest.param(lambda a: _rotation_matrix_xyz(a, a, a), 1e-3, id="small-xyz"),
        pytest.param(_rotation_matrix_x, np.pi / 2, id="large-x"),
        pytest.param(_rotation_matrix_y, np.pi * 0.75, id="large-y"),
        pytest.param(_rotation_matrix_z, np.pi * 0.9, id="large-z"),
        pytest.param(lambda a: _rotation_matrix_xyz(a, a, a), np.pi / 6, id="large-xyz"),
    ],
)
def test_compute_incremental_rotation_matrices_handles_small_and_large_rotations(rotation_builder, angle):
    """Small absolute changes should yield small deltas, and larger changes should yield larger deltas."""
    R0 = np.eye(3, dtype=np.float64)
    R1 = rotation_builder(angle)
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
def test_compute_incremental_rotation_matrices_returns_expected_shape_and_dtype(n_frames):
    """The function should return (n_frames - 1, 3, 3) float64 matrices."""
    D = _rotation_matrix_x(0.1)
    frames = [np.eye(3, dtype=np.float64)]
    for _ in range(n_frames - 1):
        frames.append(frames[-1] @ D)
    frames = np.stack(frames)

    deltas = compute_incremental_rotation_matrices(frames)
    assert deltas.shape == (n_frames - 1, 3, 3)
    assert deltas.dtype == np.float64
