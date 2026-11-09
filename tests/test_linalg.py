import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpex.linalg import (
    CholeskySystem,
    CovarianceProjection,
    DenseBackend,
    LinearSystem,
    LinearSystemUpdate,
    SolverConfiguration,
    SymmetricLinearOperator,
)


class _DenseUpdate(LinearSystemUpdate):
    system: "_ArrayOperator"

    def __init__(self, previous, cross, appended_covariance, indices):
        self.indices = indices
        self._cross = cross
        self._appended_covariance = appended_covariance
        self._previous_capacity = previous.shape[0]
        matrix = previous.at[:, indices].set(cross)
        matrix = matrix.at[indices, :].set(cross.T)
        matrix = matrix.at[indices[:, None], indices[None, :]].set(appended_covariance)
        self.system = _ArrayOperator(matrix)

    @property
    def previous_capacity(self) -> int:
        return self._previous_capacity

    def trailing_blocks(self, *, tile_size: int | None) -> tuple[jax.Array, jax.Array]:
        del tile_size
        return self._cross, self._appended_covariance


class _ArrayOperator(eqx.Module, SymmetricLinearOperator):
    matrix: jax.Array

    @property
    def capacity(self) -> int:
        return self.matrix.shape[0]

    def matmul(self, right_hand_side, *, tile_size: int = 128):
        return self.matrix @ right_hand_side

    def diagonal(self):
        return jnp.diag(self.matrix)

    def materialize(self, *, tile_size: int | None) -> jax.Array:
        del tile_size
        return self.matrix


@pytest.mark.parametrize(
    "interface",
    [
        SolverConfiguration,
        LinearSystem,
        SymmetricLinearOperator,
        LinearSystemUpdate,
        CovarianceProjection,
    ],
)
def test_incomplete_backend_implementations_cannot_be_instantiated(interface):
    incomplete = type("Incomplete", (eqx.Module, interface), {})
    with pytest.raises(TypeError, match="abstract"):
        incomplete()


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_cholesky_solves_vectors_and_matrices(compiled) -> None:
    # Both triangles contribute through the symmetric-input convention.
    matrix = jnp.array([[2.0, 0.4, -0.15], [0.2, 1.2, 0.3], [-0.05, 0.1, 1.5]])
    jitter = 1e-5
    factorize = (
        eqx.filter_jit(CholeskySystem.from_matrix)
        if compiled
        else CholeskySystem.from_matrix
    )
    factorization = factorize(matrix, jitter)
    assert isinstance(factorization, LinearSystem)
    stabilized = 0.5 * (matrix + matrix.T) + jitter * jnp.eye(3)

    vector = jnp.array([1.0, -0.5, 0.3])
    matrix_right_hand_side = jnp.array([[1.0, 2.0], [-0.5, 0.25], [0.3, -0.1]])

    def solve(factor, rhs):
        return factor.solve(rhs)

    if compiled:
        solve = eqx.filter_jit(solve)

    assert jnp.allclose(
        solve(factorization, vector),
        jnp.linalg.solve(stabilized, vector),
    )
    assert jnp.allclose(
        solve(factorization, matrix_right_hand_side),
        jnp.linalg.solve(stabilized, matrix_right_hand_side),
    )
    assert jnp.allclose(
        factorization.logdet(),
        jnp.linalg.slogdet(stabilized)[1],
    )
    quadratic = (
        eqx.filter_jit(lambda s, b: s.quadratic_form(b))
        if compiled
        else (lambda s, b: s.quadratic_form(b))
    )
    assert jnp.allclose(
        quadratic(factorization, vector),
        vector @ jnp.linalg.solve(stabilized, vector),
    )
    assert jnp.allclose(factorization.factor.factor, jnp.linalg.cholesky(stabilized))
    assert jnp.allclose(
        solve(CholeskySystem(factorization.factor, None), vector),
        solve(factorization, vector),
    )


@pytest.mark.parametrize("shape", [(2,), (1, 2)])
def test_dense_factorization_rejects_shapes_that_would_broadcast(shape):
    with pytest.raises(ValueError, match="matrix must be square"):
        CholeskySystem.from_matrix(jnp.ones(shape), jitter=1.0)


def test_covariance_reductions_match_direct_solve():
    matrix = jnp.array([[2.0, 0.3, -0.1], [0.3, 1.2, 0.2], [-0.1, 0.2, 1.5]])
    cross = jnp.array([[1.0, 0.2], [0.3, -0.8], [-0.4, 0.5]])
    system = CholeskySystem.from_matrix(matrix, jitter=0.0)
    expected = cross.T @ jnp.linalg.solve(matrix, cross)
    covariance, variance = eqx.filter_jit(
        lambda s, c: (s.covariance_reduction(c), s.variance_reduction(c))
    )(system, cross)
    assert jnp.allclose(covariance, expected, atol=1e-6, rtol=1e-5)
    assert jnp.allclose(variance, jnp.diag(expected), atol=1e-6, rtol=1e-5)
    for invalid in (jnp.ones(3), jnp.ones((2, 3))):
        with pytest.raises((ValueError, TypeError)):
            system.covariance_reduction(invalid)
        with pytest.raises((ValueError, TypeError)):
            system.variance_reduction(invalid)


def test_repeated_interspersed_appends_preserve_canonical_consumers_across_jit():
    # Two updates move different identity slots past retained observations.
    # Consuming each returned state separately catches stale inverse permutations
    # that a fused append/coefficient calculation would not expose.
    features = jnp.sin(jnp.arange(21.0).reshape(7, 3)) / 3
    covariance = features @ features.T + 0.4 * jnp.eye(7)
    active = jnp.array([False, True, False, False, True, False, False])
    rhs = jnp.linspace(-0.4, 0.7, 7)
    cross = jnp.cos(jnp.arange(28.0).reshape(7, 4))
    tolerance = 2e-5 if covariance.dtype == jnp.float32 else 1e-11

    def masked_matrix(active):
        return covariance * active[:, None] * active[None, :] + jnp.diag(~active)

    @eqx.filter_jit
    def prepare(matrix, active):
        system = DenseBackend().prepare(_ArrayOperator(matrix), active=active)
        return system, system.solve(jnp.where(active, rhs, jnp.nan))

    @eqx.filter_jit
    def append(system, solution, previous, active, indices):
        update = _DenseUpdate(
            previous,
            covariance[:, indices] * active[:, None],
            covariance[indices[:, None], indices[None, :]],
            indices,
        )
        active = active.at[indices].set(True)
        return system.append_solve(update, solution, jnp.where(active, rhs, jnp.nan))

    @eqx.filter_jit
    def consume(system, rhs, cross):
        left = system.project(cross[:, :2])
        right = system.project(cross[:, 2:])
        return (
            system.size,
            system.solve(cross.reshape(7, 2, 2)),
            system.quadratic_form(rhs),
            system.logdet(),
            system.covariance_reduction(cross),
            system.variance_reduction(cross),
            left.cross_gram(right),
        )

    system, solution = prepare(masked_matrix(active), active)
    assert system.size == 2
    assert system.capacity == 7
    for indices, size in ((jnp.array([0, 5]), 4), (jnp.array([2, 6]), 6)):
        system, solution = append(
            system, solution, masked_matrix(active), active, indices
        )
        active = active.at[indices].set(True)
        matrix = np.asarray(masked_matrix(active))
        b = jnp.where(active, rhs, 0)
        c = jnp.where(active[:, None], cross, 0)
        expected_solution = np.linalg.solve(matrix, np.asarray(b))
        solved_cross = np.linalg.solve(matrix, np.asarray(c))
        reduction = np.asarray(c).T @ solved_cross
        np.testing.assert_allclose(
            solution, expected_solution, rtol=tolerance, atol=tolerance
        )
        expected = (
            size,
            solved_cross.reshape(7, 2, 2),
            np.asarray(b) @ expected_solution,
            np.linalg.slogdet(matrix)[1],
            reduction,
            np.diag(reduction),
            reduction[:2, 2:],
        )
        consumed = consume(
            system,
            jnp.where(active, rhs, jnp.nan),
            jnp.where(active[:, None], cross, jnp.nan),
        )
        assert system.capacity == 7
        for actual, reference in zip(consumed, expected, strict=True):
            np.testing.assert_allclose(
                actual, reference, rtol=tolerance, atol=tolerance
            )


@pytest.mark.parametrize("compiled", [False, True])
def test_linear_solves_preserve_empty_rhs_batches(compiled):
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.2]])
    dense = CholeskySystem.from_matrix(matrix, jitter=0.0)
    solve = eqx.filter_jit(dense.solve) if compiled else dense.solve
    for shape in ((2, 0), (2, 3, 0)):
        assert solve(jnp.empty(shape)).shape == shape


@pytest.mark.parametrize("compiled", [False, True])
def test_dense_cholesky_returns_nans_on_failed_factorization_and_append(compiled):
    def prepare(matrix):
        backend = DenseBackend()
        assert isinstance(backend, SolverConfiguration)
        return backend.prepare(_ArrayOperator(matrix), materialization_tile_size=None)

    def factorize(matrix):
        return CholeskySystem.from_matrix(matrix, jitter=0.0)

    for operation in (prepare, factorize):
        if compiled:
            operation = eqx.filter_jit(operation)
        result = operation(jnp.array([[1.0, 2.0], [2.0, 1.0]]))
        assert jnp.any(jnp.isnan(result.factor.factor))
        assert jnp.any(jnp.isnan(result.solve(jnp.ones(2))))

    system = DenseBackend().prepare(
        _ArrayOperator(jnp.eye(2)), active=jnp.array([True, False])
    )
    for solve in (False, True):

        def append(matrix, solve=solve):
            update = _DenseUpdate(
                jnp.eye(2), jnp.array([[1.0], [0.0]]), matrix, jnp.array([1])
            )
            if solve:
                return system.append_solve(update, jnp.array([1.0, 0.0]), jnp.ones(2))
            return system.append(update)

        operation = eqx.filter_jit(append) if compiled else append
        result = operation(jnp.zeros((1, 1)))
        if isinstance(result, tuple):
            result, solution = result
            assert jnp.any(jnp.isnan(solution))
        assert jnp.any(jnp.isnan(result.factor.factor))


def test_dense_factorization_failure_returns_nan_in_its_batch_member():
    def scalar(matrix):
        return CholeskySystem.from_matrix(matrix, jitter=0.0).factor.factor[0, 0]

    valid = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    invalid = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    operation = eqx.filter_jit(jax.vmap(scalar))
    result = operation(jnp.stack((valid, invalid)))
    assert jnp.isfinite(result[0])
    assert jnp.isnan(result[1])


def test_dense_factorization_returns_batched_factors_and_active_sizes():
    matrices = jnp.array(
        [
            [[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.5]],
            [[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]],
        ]
    )
    active = jnp.array([[True, False, True], [False, False, False]])
    matrices = jnp.where(active[:, :, None] & active[:, None, :], matrices, 0)
    matrices = matrices + jax.vmap(jnp.diag)(~active)
    factorize = eqx.filter_jit(
        jax.vmap(lambda x, a: DenseBackend().prepare(_ArrayOperator(x), active=a))
    )
    result = factorize(matrices, active)
    np.testing.assert_array_equal(result.size, [2, 0])
    assert result.capacity == 3
    tolerance = 2e-6 if matrices.dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(
        result.factor.factor,
        np.linalg.cholesky(np.asarray(matrices)),
        rtol=tolerance,
        atol=tolerance,
    )


def test_dense_cholesky_solve_is_differentiable() -> None:
    right_hand_side = jnp.array([0.4, -0.2])

    def objective(log_scale: jax.Array) -> jax.Array:
        scale = jnp.exp(log_scale)
        matrix = scale * jnp.array([[1.5, 0.2], [0.2, 0.8]])
        factorization = CholeskySystem.from_matrix(matrix, jitter=1e-5)
        solution = factorization.solve(right_hand_side)
        return right_hand_side @ solution + factorization.logdet()

    gradient = jax.grad(objective)(jnp.array(0.1))
    epsilon = 1e-3
    finite_difference = (
        objective(jnp.array(0.1 + epsilon)) - objective(jnp.array(0.1 - epsilon))
    ) / (2 * epsilon)

    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, finite_difference, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("new_size", [1, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_dense_cholesky_updates_existing_solution_when_appending(
    new_size, compiled
) -> None:
    matrix = jnp.array([[2.0, 0.3], [0.3, 1.2]])
    cross = jnp.array([[0.2, -0.1], [-0.1, 0.25]])
    new_covariance = jnp.array([[1.5, 0.15], [0.15, 0.9]])
    right_hand_side = jnp.array([1.0, -0.5])
    appended_right_hand_side = jnp.array([0.3, -0.2])
    cross = cross[:, :new_size]
    new_covariance = new_covariance[:new_size, :new_size]
    appended_right_hand_side = appended_right_hand_side[:new_size]
    previous = jnp.eye(3 + new_size).at[:2, :2].set(matrix + 1e-5 * jnp.eye(2))
    factorization = DenseBackend().prepare(
        _ArrayOperator(previous),
        active=jnp.arange(3 + new_size) < 2,
    )
    solution = factorization.solve(jnp.pad(right_hand_side, (0, 1 + new_size)))
    update = _DenseUpdate(
        previous,
        jnp.pad(cross, ((0, 1 + new_size), (0, 0))),
        new_covariance + 1e-5 * jnp.eye(new_size),
        jnp.arange(2, 2 + new_size),
    )

    append_solve = (
        eqx.filter_jit(factorization.append_solve)
        if compiled
        else factorization.append_solve
    )
    appended, updated_solution = append_solve(
        update,
        solution,
        jnp.concatenate((right_hand_side, appended_right_hand_side, jnp.zeros(1))),
    )
    complete_right_hand_side = jnp.concatenate(
        (right_hand_side, appended_right_hand_side, jnp.zeros(1))
    )

    assert jnp.allclose(
        updated_solution,
        jnp.linalg.solve(
            update.system.matrix,
            complete_right_hand_side,
        ),
        rtol=2e-6,
        atol=2e-6,
    )
    assert jnp.allclose(
        appended.solve(complete_right_hand_side), updated_solution, rtol=2e-6, atol=2e-6
    )
    assert jnp.allclose(
        appended.quadratic_form(complete_right_hand_side),
        complete_right_hand_side @ updated_solution,
        rtol=2e-6,
        atol=2e-6,
    )
    assert jnp.allclose(
        appended.logdet(),
        jnp.linalg.slogdet(update.system.matrix)[1],
        rtol=2e-6,
        atol=2e-6,
    )
