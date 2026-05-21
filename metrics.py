"""Metrics for continuous-outcome uplift modeling."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc


def _sorted_arrays(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    uplift = np.asarray(uplift, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int8)
    if len(y_true) != len(uplift) or len(y_true) != len(treatment):
        raise ValueError("y_true, uplift and treatment must have the same length")
    order = np.argsort(uplift, kind="mergesort")[::-1]
    return y_true[order], uplift[order], treatment[order]


def _safe_group_diff(y: np.ndarray, treatment: np.ndarray) -> float:
    treatment = np.asarray(treatment, dtype=np.int8)
    treated = treatment == 1
    control = treatment == 0
    if treated.sum() == 0 or control.sum() == 0:
        return 0.0
    return float(np.mean(y[treated]) - np.mean(y[control]))


def uplift_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative spend uplift curve sorted by predicted uplift score."""
    y_sorted, _, t_sorted = _sorted_arrays(y_true, uplift, treatment)
    n = len(y_sorted)
    xs = np.arange(1, n + 1)
    treated = t_sorted == 1
    control = t_sorted == 0

    treated_count = np.cumsum(treated)
    control_count = np.cumsum(control)
    treated_sum = np.cumsum(np.where(treated, y_sorted, 0.0))
    control_sum = np.cumsum(np.where(control, y_sorted, 0.0))

    treated_mean = np.divide(treated_sum, treated_count, out=np.zeros(n), where=treated_count > 0)
    control_mean = np.divide(control_sum, control_count, out=np.zeros(n), where=control_count > 0)
    curve = (treated_mean - control_mean) * xs
    return np.r_[0, xs], np.r_[0.0, curve]


def qini_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous-outcome Qini curve."""
    y_sorted, _, t_sorted = _sorted_arrays(y_true, uplift, treatment)
    n = len(y_sorted)
    xs = np.arange(1, n + 1)
    treated = t_sorted == 1
    control = t_sorted == 0

    treated_count = np.cumsum(treated)
    control_count = np.cumsum(control)
    treated_sum = np.cumsum(np.where(treated, y_sorted, 0.0))
    control_sum = np.cumsum(np.where(control, y_sorted, 0.0))

    expected_control = control_sum * np.divide(
        treated_count,
        control_count,
        out=np.zeros(n),
        where=control_count > 0,
    )
    return np.r_[0, xs], np.r_[0.0, treated_sum - expected_control]


def auuc_score(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    x, y = uplift_curve(y_true, uplift, treatment)
    if len(x) <= 1:
        return 0.0
    return float(auc(x / x[-1], y / max(len(y_true), 1)))


def qini_auc_score(y_true: np.ndarray, uplift: np.ndarray, treatment: np.ndarray) -> float:
    x, y = qini_curve(y_true, uplift, treatment)
    if len(x) <= 1:
        return 0.0
    return float(auc(x / x[-1], y / max(len(y_true), 1)))


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float,
) -> float:
    n_top = max(1, int(np.ceil(len(uplift) * k)))
    y_sorted, _, t_sorted = _sorted_arrays(y_true, uplift, treatment)
    return _safe_group_diff(y_sorted[:n_top], t_sorted[:n_top])


def bootstrap_uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float,
    *,
    n_iterations: int,
    ci: float,
    random_state: int,
) -> dict[str, float]:
    n_top = max(1, int(np.ceil(len(uplift) * k)))
    y_sorted, _, t_sorted = _sorted_arrays(y_true, uplift, treatment)
    y_top = y_sorted[:n_top]
    t_top = t_sorted[:n_top]

    rng = np.random.default_rng(random_state)
    scores = np.empty(n_iterations, dtype=float)
    for i in range(n_iterations):
        idx = rng.integers(0, n_top, size=n_top)
        scores[i] = _safe_group_diff(y_top[idx], t_top[idx])

    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(np.quantile(scores, alpha)),
        "ci_upper": float(np.quantile(scores, 1.0 - alpha)),
    }


def average_uplift(y_true: np.ndarray, treatment: np.ndarray) -> float:
    return _safe_group_diff(np.asarray(y_true, dtype=float), np.asarray(treatment, dtype=np.int8))


def irr_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float,
) -> float:
    base = average_uplift(y_true, treatment)
    if abs(base) < 1e-12:
        return 0.0
    return uplift_at_k(y_true, uplift, treatment, k) / base


def evaluate_all_metrics(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    *,
    k: float,
    k_grid: tuple[float, ...] | None = None,
    bootstrap_iterations: int = 200,
    bootstrap_ci: float = 0.80,
    random_state: int = 42,
) -> dict[str, float]:
    result = {
        "auuc": auuc_score(y_true, uplift, treatment),
        "qini": qini_auc_score(y_true, uplift, treatment),
        "avg_uplift": average_uplift(y_true, treatment),
    }

    for kk in k_grid or (k,):
        pct = int(round(kk * 100))
        result[f"uplift_at_{pct}pct"] = uplift_at_k(y_true, uplift, treatment, kk)
        result[f"irr_at_{pct}pct"] = irr_score(y_true, uplift, treatment, kk)

    main_pct = int(round(k * 100))
    boot = bootstrap_uplift_at_k(
        y_true,
        uplift,
        treatment,
        k,
        n_iterations=bootstrap_iterations,
        ci=bootstrap_ci,
        random_state=random_state,
    )
    result[f"uplift_at_{main_pct}pct_bootstrap_mean"] = boot["mean"]
    result[f"uplift_at_{main_pct}pct_ci_lower"] = boot["ci_lower"]
    result[f"uplift_at_{main_pct}pct_ci_upper"] = boot["ci_upper"]
    result[f"uplift_at_{main_pct}pct_bootstrap_std"] = boot["std"]
    return result
