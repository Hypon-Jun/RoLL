from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import svd


SCAD_A = 3.7


def scad_penalty_term(
    values: np.ndarray,
    lambda_value: float,
    a: float = SCAD_A,
) -> float:
    absolute = np.abs(np.asarray(values, dtype=float))
    penalty = np.zeros_like(absolute)

    first = absolute <= lambda_value
    middle = (absolute > lambda_value) & (absolute <= a * lambda_value)
    last = absolute > a * lambda_value

    penalty[first] = lambda_value * absolute[first]
    penalty[middle] = (
        2.0 * a * lambda_value * absolute[middle]
        - absolute[middle] ** 2
        - lambda_value**2
    ) / (2.0 * (a - 1.0))
    penalty[last] = (a + 1.0) * lambda_value**2 / 2.0
    return float(np.sum(penalty))


def scad_derivative(
    values: np.ndarray,
    lambda_value: float,
    a: float = SCAD_A,
) -> np.ndarray:
    absolute = np.abs(np.asarray(values, dtype=float))
    derivative = np.zeros_like(absolute)
    first = absolute <= lambda_value
    middle = (absolute > lambda_value) & (absolute <= a * lambda_value)
    derivative[first] = lambda_value
    derivative[middle] = (
        a * lambda_value - absolute[middle]
    ) / (a - 1.0)
    return derivative


def _prox_scad_values(
    values: np.ndarray,
    lambda_value: float,
    curvature: float,
    a: float = SCAD_A,
) -> np.ndarray:
    if curvature <= 0.0:
        raise ValueError("curvature must be positive")
    values = np.asarray(values, dtype=float)
    absolute = np.abs(values)
    result = np.zeros_like(values)

    threshold = lambda_value / curvature
    boundary_1 = lambda_value * (1.0 + 1.0 / curvature)
    boundary_2 = a * lambda_value

    first = absolute <= boundary_1
    result[first] = np.sign(values[first]) * np.maximum(
        absolute[first] - threshold,
        0.0,
    )

    middle = (absolute > boundary_1) & (absolute <= boundary_2)
    denominator = a - 1.0 - 1.0 / curvature
    if denominator <= 1e-8:
        result[middle] = values[middle]
    else:
        result[middle] = (
            (a - 1.0) * values[middle]
            - np.sign(values[middle]) * a * threshold
        ) / denominator

    last = absolute > boundary_2
    result[last] = values[last]
    return result


def prox_B_scad(
    matrix: np.ndarray,
    lambda_value: float,
    curvature: float,
    a: float = SCAD_A,
) -> np.ndarray:
    U, singular_values, Vt = svd(matrix, full_matrices=False)
    new_values = _prox_scad_values(
        singular_values,
        lambda_value,
        curvature,
        a,
    )
    return (U * new_values[None, :]) @ Vt


def prox_group_scad(
    matrix: np.ndarray,
    lambda_value: float,
    curvature: float,
    a: float = SCAD_A,
) -> np.ndarray:
    row_norms = np.linalg.norm(matrix, axis=1)
    new_norms = _prox_scad_values(
        row_norms,
        lambda_value,
        curvature,
        a,
    )
    scale = np.zeros_like(row_norms)
    nonzero = row_norms > 0.0
    scale[nonzero] = new_norms[nonzero] / row_norms[nonzero]
    return matrix * scale[:, None]


def scad_nuclear(
    matrix: np.ndarray,
    lambda_value: float,
    a: float = SCAD_A,
) -> float:
    return scad_penalty_term(
        svd(matrix, full_matrices=False, compute_uv=False),
        lambda_value,
        a,
    )


def scad_l21(
    matrix: np.ndarray,
    lambda_value: float,
    a: float = SCAD_A,
) -> float:
    return scad_penalty_term(
        np.linalg.norm(matrix, axis=1),
        lambda_value,
        a,
    )


def loss_function_without_penalty(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> float:
    residual = X @ B + C - Y
    return 0.5 * float(np.sum(residual * residual))


def loss_function(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    lambda_B: float,
    lambda_C: float,
) -> float:
    return (
        loss_function_without_penalty(X, Y, B, C)
        + scad_nuclear(B, lambda_B)
        + scad_l21(C, lambda_C)
    )


def create_objective_function(X: np.ndarray, Y: np.ndarray):
    def objective(B: np.ndarray, C: np.ndarray) -> float:
        return loss_function_without_penalty(X, Y, B, C)

    return objective


def grad_of_B(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    return X.T @ (X @ B + C - Y)


def grad_of_C(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> np.ndarray:
    return X @ B + C - Y


def inverse_learning_rate(X: np.ndarray) -> float:
    largest = float(svd(X, full_matrices=False, compute_uv=False)[0])
    return largest**2 + 1.0


@dataclass(frozen=True)
class LineSearchConfig:
    start_iter: int = 50
    search_interval: int = 10
    M: int = 2
    lower_c: float = 0.7
    upper_c: float = 2.0
    relative_band_c: float = 0.22
    seed_mu: float = 1e-3
    r_tolerance: float = 0.0

    def validate(self) -> None:
        if self.start_iter < 0:
            raise ValueError("start_iter must be non-negative")
        if self.search_interval < 1:
            raise ValueError("search_interval must be positive")
        if self.M < 1:
            raise ValueError("M must be positive")
        if not 0.0 < self.lower_c < 1.0:
            raise ValueError("lower_c must lie in (0, 1)")
        if self.upper_c <= 1.0:
            raise ValueError("upper_c must be greater than one")
        if not 0.0 <= self.relative_band_c < 1.0:
            raise ValueError("relative_band_c must lie in [0, 1)")
        if self.seed_mu <= 0.0:
            raise ValueError("seed_mu must be positive")
        if self.r_tolerance < 0.0:
            raise ValueError("r_tolerance must be non-negative")


@dataclass(frozen=True)
class CandidateState:
    B: np.ndarray
    C: np.ndarray
    residual: np.ndarray
    smooth_loss: float
    penalty_B: float
    penalty_C: float
    full_objective: float
    singular_values: np.ndarray
    svd_U: np.ndarray
    svd_Vt: np.ndarray
    row_norms: np.ndarray


@dataclass(frozen=True)
class RapidResult:
    B: np.ndarray
    C: np.ndarray
    last_B: np.ndarray
    last_C: np.ndarray
    best_full_B: np.ndarray
    best_full_C: np.ndarray
    iterations: int
    elapsed_time: float
    best_objective: float
    last_objective: float
    best_iteration: int
    converged: bool
    history: List[Dict[str, object]]


class CandidateEvaluation(NamedTuple):
    state: CandidateState
    gamma_B: np.ndarray
    gamma_C: np.ndarray
    theta: float
    mu: float
    rho: float
    W_value: float
    R_value: float


class LineSearchOutcome(NamedTuple):
    selected: CandidateEvaluation
    number_of_trials: int
    W_max: float
    eligible_count: int
    fallback: bool


def _validate_problem(
    X: np.ndarray,
    Y: np.ndarray,
    B_init: np.ndarray,
    C_init: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    B_init = np.asarray(B_init, dtype=float)
    C_init = np.asarray(C_init, dtype=float)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be two-dimensional")
    n, p = X.shape
    if Y.shape[0] != n:
        raise ValueError("X and Y must have the same number of rows")
    m = Y.shape[1]
    if B_init.shape != (p, m):
        raise ValueError("B_init must have shape (p, m)")
    if C_init.shape != (n, m):
        raise ValueError("C_init must have shape (n, m)")
    if not all(np.all(np.isfinite(value)) for value in (X, Y, B_init, C_init)):
        raise ValueError("all inputs must be finite")
    return X, Y, B_init, C_init


def _make_state(
    X: np.ndarray,
    Y: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    lambda_B: float,
    lambda_C: float,
    U: Optional[np.ndarray] = None,
    singular_values: Optional[np.ndarray] = None,
    Vt: Optional[np.ndarray] = None,
    row_norms: Optional[np.ndarray] = None,
) -> CandidateState:
    if singular_values is None or U is None or Vt is None:
        U, singular_values, Vt = svd(B, full_matrices=False)
    if row_norms is None:
        row_norms = np.linalg.norm(C, axis=1)
    residual = X @ B + C - Y
    smooth = 0.5 * float(np.sum(residual * residual))
    penalty_B = scad_penalty_term(singular_values, lambda_B)
    penalty_C = scad_penalty_term(row_norms, lambda_C)
    return CandidateState(
        B=B,
        C=C,
        residual=residual,
        smooth_loss=smooth,
        penalty_B=penalty_B,
        penalty_C=penalty_C,
        full_objective=smooth + penalty_B + penalty_C,
        singular_values=singular_values,
        svd_U=U,
        svd_Vt=Vt,
        row_norms=row_norms,
    )


def _prox_B_scad_cached(
    matrix: np.ndarray,
    lambda_value: float,
    curvature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    U, singular_values, Vt = svd(matrix, full_matrices=False)
    new_values = _prox_scad_values(
        singular_values,
        lambda_value,
        curvature,
    )
    B = (U * new_values[None, :]) @ Vt
    return B, U, new_values, Vt


def _prox_group_scad_cached(
    matrix: np.ndarray,
    lambda_value: float,
    curvature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    row_norms = np.linalg.norm(matrix, axis=1)
    new_norms = _prox_scad_values(
        row_norms,
        lambda_value,
        curvature,
    )
    scale = np.zeros_like(row_norms)
    nonzero = row_norms > 0.0
    scale[nonzero] = new_norms[nonzero] / row_norms[nonzero]
    return matrix * scale[:, None], new_norms


def theta_update(
    theta_previous: float,
    mu_previous: float,
    rho_previous: float,
    rho_new: float,
) -> float:
    if not np.isfinite(rho_new) or rho_new <= 0.0:
        raise FloatingPointError("rho must be finite and positive")
    previous = theta_previous * (
        rho_previous * theta_previous + mu_previous
    )
    radicand = previous**2 + 4.0 * rho_new * previous
    scale = max(1.0, abs(previous**2), abs(4.0 * rho_new * previous))
    if radicand < -1e-14 * scale:
        raise FloatingPointError("theta_update received a negative radicand")
    theta_new = (
        -previous + math.sqrt(max(0.0, radicand))
    ) / (2.0 * rho_new)
    if not np.isfinite(theta_new) or theta_new <= 0.0:
        raise FloatingPointError("theta_update produced an invalid value")
    return float(theta_new)


def gamma_update(
    beta: np.ndarray,
    beta_previous: np.ndarray,
    theta_new: float,
    theta_previous: float,
    rho_previous: float,
    mu_previous: float,
) -> np.ndarray:
    denominator = rho_previous * theta_previous + mu_previous
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("gamma_update denominator must be positive")
    momentum = (
        rho_previous
        * theta_new
        * (1.0 - theta_previous)
        / denominator
    )
    gamma = beta + momentum * (beta - beta_previous)
    if not np.all(np.isfinite(gamma)):
        raise FloatingPointError("gamma_update produced NaN or Inf")
    return gamma


def _proximal_step(
    X: np.ndarray,
    Y: np.ndarray,
    gamma_B: np.ndarray,
    gamma_C: np.ndarray,
    L: float,
    lambda_B: float,
    lambda_C: float,
    need_penalty_gradient: bool = False,
) -> CandidateState:
    residual = X @ gamma_B + gamma_C - Y
    B_argument = gamma_B - (X.T @ residual) / L
    C_argument = gamma_C - residual / L

    B, U_argument, new_singular_values, Vt_argument = _prox_B_scad_cached(
        B_argument,
        lambda_B,
        L,
    )
    C, row_norms = _prox_group_scad_cached(
        C_argument,
        lambda_C,
        L,
    )

    if need_penalty_gradient:
        U, singular_values, Vt = svd(B, full_matrices=False)
    else:
        U, singular_values, Vt = (
            U_argument,
            new_singular_values,
            Vt_argument,
        )
    return _make_state(
        X,
        Y,
        B,
        C,
        lambda_B,
        lambda_C,
        U=U,
        singular_values=singular_values,
        Vt=Vt,
        row_norms=row_norms,
    )


def _relative_changes(
    new_state: CandidateState,
    old_state: CandidateState,
) -> Tuple[float, float, float]:
    rel_B = float(
        np.linalg.norm(new_state.B - old_state.B)
        / max(1.0, float(np.linalg.norm(old_state.B)))
    )
    rel_C = float(
        np.linalg.norm(new_state.C - old_state.C)
        / max(1.0, float(np.linalg.norm(old_state.C)))
    )
    rel_objective = float(
        abs(new_state.full_objective - old_state.full_objective)
        / max(1.0, abs(old_state.full_objective))
    )
    return rel_B, rel_C, rel_objective


def _bregman_psi_quadratic(
    X: np.ndarray,
    first_B: np.ndarray,
    first_C: np.ndarray,
    second_B: np.ndarray,
    second_C: np.ndarray,
    mu: float,
) -> float:
    delta_B = first_B - second_B
    delta_C = first_C - second_C
    prediction_delta = X @ delta_B + delta_C
    smooth_bregman = 0.5 * float(
        np.sum(prediction_delta * prediction_delta)
    )
    l2_bregman = 0.5 * float(
        np.sum(delta_B * delta_B) + np.sum(delta_C * delta_C)
    )
    return smooth_bregman - mu * l2_bregman


def _penalty_gradient_inner(
    state: CandidateState,
    target_B: np.ndarray,
    target_C: np.ndarray,
    lambda_B: float,
    lambda_C: float,
) -> float:
    derivative_B = scad_derivative(state.singular_values, lambda_B)
    gradient_B = (
        state.svd_U * derivative_B[None, :]
    ) @ state.svd_Vt

    derivative_C = scad_derivative(state.row_norms, lambda_C)
    row_scale = np.zeros_like(state.row_norms)
    nonzero = state.row_norms > 0.0
    row_scale[nonzero] = (
        derivative_C[nonzero] / state.row_norms[nonzero]
    )
    gradient_C = state.C * row_scale[:, None]
    return float(
        np.sum(gradient_B * (target_B - state.B))
        + np.sum(gradient_C * (target_C - state.C))
    )


def _candidate_RE_criterion(
    X: np.ndarray,
    optimal_state: CandidateState,
    gamma_B: np.ndarray,
    gamma_C: np.ndarray,
    beta_state: CandidateState,
    candidate_state: CandidateState,
    theta: float,
    mu: float,
    rho: float,
    lambda_B: float,
    lambda_C: float,
) -> Tuple[float, float]:
    if theta <= 0.0:
        raise FloatingPointError("theta must be positive")

    dpsi_new = _bregman_psi_quadratic(
        X,
        candidate_state.B,
        candidate_state.C,
        gamma_B,
        gamma_C,
        mu,
    )
    dpsi_beta = _bregman_psi_quadratic(
        X,
        beta_state.B,
        beta_state.C,
        gamma_B,
        gamma_C,
        mu,
    )
    delta_B = candidate_state.B - gamma_B
    delta_C = candidate_state.C - gamma_C
    l2_new = 0.5 * float(
        np.sum(delta_B * delta_B) + np.sum(delta_C * delta_C)
    )
    R_value = (
        rho * l2_new
        - dpsi_new
        + (1.0 - theta) * dpsi_beta
    )

    dpsi_optimal = _bregman_psi_quadratic(
        X,
        optimal_state.B,
        optimal_state.C,
        gamma_B,
        gamma_C,
        mu,
    )
    penalty_optimal = optimal_state.penalty_B + optimal_state.penalty_C
    penalty_beta = beta_state.penalty_B + beta_state.penalty_C
    penalty_new = candidate_state.penalty_B + candidate_state.penalty_C
    mixture_B = theta * optimal_state.B + (1.0 - theta) * beta_state.B
    mixture_C = theta * optimal_state.C + (1.0 - theta) * beta_state.C
    gradient_inner = _penalty_gradient_inner(
        candidate_state,
        mixture_B,
        mixture_C,
        lambda_B,
        lambda_C,
    )
    E_value = (
        dpsi_optimal
        + penalty_optimal
        + ((1.0 - theta) / theta) * penalty_beta
        - penalty_new / theta
        - gradient_inner / theta
    )
    W_value = R_value + theta * E_value
    return float(W_value), float(R_value)


def _candidate_for_mu(
    X: np.ndarray,
    Y: np.ndarray,
    optimal_state: CandidateState,
    beta_state: CandidateState,
    beta_previous_state: CandidateState,
    theta_previous: float,
    mu_previous: float,
    rho_previous: float,
    mu_try: float,
    L: float,
    lambda_B: float,
    lambda_C: float,
) -> CandidateEvaluation:
    if not np.isfinite(mu_try) or not 0.0 <= mu_try < L:
        raise FloatingPointError("candidate mu must satisfy 0 <= mu < L")
    rho_try = L - mu_try
    theta_try = theta_update(
        theta_previous,
        mu_previous,
        rho_previous,
        rho_try,
    )
    gamma_B_try = gamma_update(
        beta_state.B,
        beta_previous_state.B,
        theta_try,
        theta_previous,
        rho_previous,
        mu_previous,
    )
    gamma_C_try = gamma_update(
        beta_state.C,
        beta_previous_state.C,
        theta_try,
        theta_previous,
        rho_previous,
        mu_previous,
    )
    candidate_state = _proximal_step(
        X,
        Y,
        gamma_B_try,
        gamma_C_try,
        L,
        lambda_B,
        lambda_C,
        need_penalty_gradient=True,
    )
    W_value, R_value = _candidate_RE_criterion(
        X,
        optimal_state,
        gamma_B_try,
        gamma_C_try,
        beta_state,
        candidate_state,
        theta_try,
        mu_try,
        rho_try,
        lambda_B,
        lambda_C,
    )
    if not np.isfinite(W_value) or not np.isfinite(R_value):
        raise FloatingPointError("line-search criterion became NaN or Inf")
    return CandidateEvaluation(
        state=candidate_state,
        gamma_B=gamma_B_try,
        gamma_C=gamma_C_try,
        theta=theta_try,
        mu=float(mu_try),
        rho=float(rho_try),
        W_value=W_value,
        R_value=R_value,
    )


def line_search_mu(
    X: np.ndarray,
    Y: np.ndarray,
    optimal_state: CandidateState,
    beta_state: CandidateState,
    beta_previous_state: CandidateState,
    theta: float,
    mu: float,
    rho: float,
    L: float,
    lambda_B: float,
    lambda_C: float,
    config: Optional[LineSearchConfig] = None,
) -> LineSearchOutcome:
    config = LineSearchConfig() if config is None else config
    config.validate()
    mu_0 = config.lower_c * mu if mu > 0.0 else config.seed_mu

    grid: List[float] = []
    for exponent in range(config.M):
        mu_try = float(mu_0 * config.upper_c**exponent)
        if 0.0 <= mu_try < L and (
            not grid
            or mu_try > grid[-1] * (1.0 + np.finfo(float).eps)
        ):
            grid.append(mu_try)
    if not grid:
        grid = [0.0]

    candidates = [
        _candidate_for_mu(
            X,
            Y,
            optimal_state,
            beta_state,
            beta_previous_state,
            theta,
            mu,
            rho,
            mu_try,
            L,
            lambda_B,
            lambda_C,
        )
        for mu_try in grid
    ]

    W_values = np.asarray(
        [candidate.W_value for candidate in candidates],
        dtype=float,
    )
    W_max = float(np.max(W_values))
    if W_max > 0.0:
        lower = (1.0 - config.relative_band_c) * W_max
        in_band = (W_values >= lower) & (W_values <= W_max)
    else:
        in_band = W_values == W_max

    feasible = np.asarray(
        [
            candidate.R_value >= -config.r_tolerance
            and 0.0 <= candidate.mu < L
            and candidate.rho > 0.0
            for candidate in candidates
        ],
        dtype=bool,
    )
    eligible = np.flatnonzero(in_band & feasible)
    if eligible.size:
        selected_index = max(
            (int(index) for index in eligible),
            key=lambda index: candidates[index].mu,
        )
        return LineSearchOutcome(
            selected=candidates[selected_index],
            number_of_trials=len(candidates),
            W_max=W_max,
            eligible_count=int(eligible.size),
            fallback=False,
        )

    fallback = _candidate_for_mu(
        X,
        Y,
        optimal_state,
        beta_state,
        beta_previous_state,
        theta,
        mu,
        rho,
        0.0,
        L,
        lambda_B,
        lambda_C,
    )
    return LineSearchOutcome(
        selected=fallback,
        number_of_trials=len(candidates),
        W_max=W_max,
        eligible_count=0,
        fallback=True,
    )


def line_search(*args, **kwargs) -> LineSearchOutcome:
    return line_search_mu(*args, **kwargs)


def _solve_rapid(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_B: float,
    lambda_C: float,
    B_init: np.ndarray,
    C_init: np.ndarray,
    max_iter: int,
    parameter_tol: float,
    objective_tol: float,
    search_config: Optional[LineSearchConfig],
    return_best_iterate: bool,
    verbose: bool,
) -> RapidResult:
    X, Y, B_init, C_init = _validate_problem(X, Y, B_init, C_init)
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    if parameter_tol < 0.0 or objective_tol < 0.0:
        raise ValueError("tolerances must be non-negative")
    if lambda_B < 0.0 or lambda_C < 0.0:
        raise ValueError("penalties must be non-negative")
    if search_config is not None:
        search_config.validate()

    started = time.perf_counter()
    L = inverse_learning_rate(X)
    if not np.isfinite(L) or L <= 0.0:
        raise FloatingPointError("L must be finite and positive")

    state = _make_state(
        X,
        Y,
        B_init.copy(),
        C_init.copy(),
        lambda_B,
        lambda_C,
    )
    best_state = state
    best_iteration = 0
    

    mu = 0.0
    rho = L
    theta = 1.0
    gamma_B = state.B.copy()
    gamma_C = state.C.copy()
    prefetched_state: Optional[CandidateState] = None
    history: List[Dict[str, object]] = []
    converged = False

    for iteration in range(max_iter):
        old_state = state
        mu_used = mu
        rho_used = rho
        theta_used = theta
        if prefetched_state is None:
            state = _proximal_step(
                X,
                Y,
                gamma_B,
                gamma_C,
                L,
                lambda_B,
                lambda_C,
            )
        else:
            state = prefetched_state
            prefetched_state = None

        if not np.isfinite(state.full_objective):
            raise FloatingPointError("objective became NaN or Inf")
        rel_B, rel_C, rel_objective = _relative_changes(state, old_state)

        if state.full_objective < best_state.full_objective:
            best_state = state
            best_iteration = iteration + 1

        

        stop_now = (
            max(rel_B, rel_C) < parameter_tol
            and rel_objective < objective_tol
        )

        triggered = False
        trials = 0
        selected_W = float("nan")
        selected_R = float("nan")
        W_max = float("nan")
        eligible_count = 0
        fallback = False

        completed_updates = iteration + 1

        if not stop_now:
            next_mu = mu
            next_rho = rho

	        scheduled = (
    		    search_config is not None
    		    and completed_updates >= search_config.start_iter
    		    and (
       		    (completed_updates - search_config.start_iter)
       		    % search_config.search_interval
       		    == 0
                )
            )

            if search_config is not None and scheduled:
                outcome = line_search_mu(
    X,
    Y,
    best_state,
    state,
    old_state,
    theta,
    mu,
    rho,
    L,
    lambda_B,
    lambda_C,
    search_config,
)
                selected = outcome.selected
                next_mu = selected.mu
                next_rho = selected.rho
                next_theta = selected.theta
                next_gamma_B = selected.gamma_B
                next_gamma_C = selected.gamma_C
                prefetched_state = selected.state
                triggered = True
                trials = outcome.number_of_trials
                selected_W = selected.W_value
                selected_R = selected.R_value
                W_max = outcome.W_max
                eligible_count = outcome.eligible_count
                fallback = outcome.fallback
            else:
                next_theta = theta_update(theta, mu, rho, next_rho)
                next_gamma_B = gamma_update(
                    state.B,
                    old_state.B,
                    next_theta,
                    theta,
                    rho,
                    mu,
                )
                next_gamma_C = gamma_update(
                    state.C,
                    old_state.C,
                    next_theta,
                    theta,
                    rho,
                    mu,
                )

            mu, rho, theta = next_mu, next_rho, next_theta
            gamma_B, gamma_C = next_gamma_B, next_gamma_C

        if not (0.0 <= mu < L and rho > 0.0):
            raise FloatingPointError("accepted mu/rho violate the domain")

        history.append(
            {
                "iteration": iteration + 1,
                "elapsed_time": time.perf_counter() - started,
                "smooth_loss": state.smooth_loss,
                "penalty_B": state.penalty_B,
                "penalty_C": state.penalty_C,
                "full_objective": state.full_objective,
                "best_full_objective": best_state.full_objective,
                "mu": mu_used,
                "rho": rho_used,
                "theta": theta_used,
                "next_mu": mu,
                "next_rho": rho,
                "next_theta": theta,
                "line_search_triggered": triggered,
                "number_of_trials": trials,
                "selected_criterion": selected_W,
                "selected_R": selected_R,
                "W_max": W_max,
                "eligible_count": eligible_count,
                "fallback_flag": fallback,
                "restart_flag": False,
                "rel_B": rel_B,
                "rel_C": rel_C,
                "rel_objective": rel_objective,
            }
        )

        if verbose:
            print(
                "iter={}, objective={:.10e}, mu={:.6e}, rho={:.6e}, "
                "rel_B={:.3e}, rel_C={:.3e}, rel_obj={:.3e}".format(
                    iteration + 1,
                    state.full_objective,
                    mu_used,
                    rho_used,
                    rel_B,
                    rel_C,
                    rel_objective,
                )
            )
        if stop_now:
            converged = True
            break

    elapsed = time.perf_counter() - started
    chosen = best_state if return_best_iterate else state
    return RapidResult(
        B=chosen.B.copy(),
        C=chosen.C.copy(),
        last_B=state.B.copy(),
        last_C=state.C.copy(),
        best_full_B=best_state.B.copy(),
        best_full_C=best_state.C.copy(),
        iterations=len(history),
        elapsed_time=elapsed,
        best_objective=best_state.full_objective,
        last_objective=state.full_objective,
        best_iteration=best_iteration,
        converged=converged,
        history=history,
    )


def acc_robust_low_rank_learning(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_B: float,
    lambda_C: float,
    B_init: np.ndarray,
    C_init: np.ndarray,
    max_iter: int = 100000,
    tol: float = 1e-6,
    objective_tol: Optional[float] = None,
    return_best_iterate: bool = False,
    return_diagnostics: bool = False,
    verbose: bool = False,
):
    result = _solve_rapid(
        X,
        Y,
        lambda_B,
        lambda_C,
        B_init,
        C_init,
        max_iter,
        tol,
        tol if objective_tol is None else objective_tol,
        search_config=None,
        return_best_iterate=return_best_iterate,
        verbose=verbose,
    )
    if return_diagnostics:
        return result
    return result.B, result.C


def acc_robust_low_rank_learning_with_line_search(
    X: np.ndarray,
    Y: np.ndarray,
    lambda_B: float,
    lambda_C: float,
    B_init: np.ndarray,
    C_init: np.ndarray,
    max_iter: int = 100000,
    tol: float = 1e-6,
    objective_tol: Optional[float] = None,
    return_best_iterate: bool = False,
    line_search_config: Optional[LineSearchConfig] = None,
    return_diagnostics: bool = False,
    verbose: bool = False,
):
    config = (
        LineSearchConfig()
        if line_search_config is None
        else line_search_config
    )
    result = _solve_rapid(
        X,
        Y,
        lambda_B,
        lambda_C,
        B_init,
        C_init,
        max_iter,
        tol,
        tol if objective_tol is None else objective_tol,
        search_config=config,
        return_best_iterate=return_best_iterate,
        verbose=verbose,
    )
    if return_diagnostics:
        return result
    return result.B, result.C


__all__: Sequence[str] = (
    "LineSearchConfig",
    "RapidResult",
    "acc_robust_low_rank_learning",
    "acc_robust_low_rank_learning_with_line_search",
    "create_objective_function",
    "grad_of_B",
    "grad_of_C",
    "inverse_learning_rate",
    "line_search",
    "line_search_mu",
    "loss_function",
    "loss_function_without_penalty",
    "prox_B_scad",
    "prox_group_scad",
    "scad_l21",
    "scad_nuclear",
)
