"""
Метрики uplift-моделирования (совместимы со scikit-uplift).

AUUC и Qini — нормализованные коэффициенты в диапазоне [0, 1]:
  0 — не лучше случайного таргетинга, 1 — идеальная модель.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc


def _cumsum(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x, dtype=np.float64)


def _check_binary(name: str, arr: np.ndarray) -> None:
    uniq = np.unique(arr)
    if not np.isin(uniq, [0, 1]).all():
        raise ValueError(f"{name} должен содержать только 0 и 1, получено: {uniq}")


def uplift_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Кривая uplift: накопленный прирост при ранжировании по score."""
    y_true = np.asarray(y_true, dtype=np.float64)
    uplift = np.asarray(uplift, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int32)
    _check_binary("treatment", treatment)
    _check_binary("y_true", y_true)

    order = np.argsort(uplift, kind="mergesort")[::-1]
    y_true = y_true[order]
    uplift = uplift[order]
    treatment = treatment[order]

    y_ctrl = y_true.copy()
    y_trmnt = y_true.copy()
    y_ctrl[treatment == 1] = 0
    y_trmnt[treatment == 0] = 0

    distinct_idx = np.where(np.diff(uplift))[0]
    thresholds = np.r_[distinct_idx, uplift.size - 1]

    num_trmnt = _cumsum(treatment)[thresholds]
    y_trmnt_sum = _cumsum(y_trmnt)[thresholds]
    num_all = thresholds + 1
    num_ctrl = num_all - num_trmnt
    y_ctrl_sum = _cumsum(y_ctrl)[thresholds]

    rate_trmnt = np.divide(y_trmnt_sum, num_trmnt, out=np.zeros_like(y_trmnt_sum), where=num_trmnt != 0)
    rate_ctrl = np.divide(y_ctrl_sum, num_ctrl, out=np.zeros_like(y_ctrl_sum), where=num_ctrl != 0)
    curve = (rate_trmnt - rate_ctrl) * num_all

    if num_all.size == 0 or curve[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve = np.r_[0, curve]

    return num_all, curve


def qini_curve(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Кривая Qini."""
    y_true = np.asarray(y_true, dtype=np.float64)
    uplift = np.asarray(uplift, dtype=np.float64)
    treatment = np.asarray(treatment, dtype=np.int32)

    order = np.argsort(uplift, kind="mergesort")[::-1]
    y_true = y_true[order]
    uplift = uplift[order]
    treatment = treatment[order]

    y_ctrl = y_true.copy()
    y_trmnt = y_true.copy()
    y_ctrl[treatment == 1] = 0
    y_trmnt[treatment == 0] = 0

    distinct_idx = np.where(np.diff(uplift))[0]
    thresholds = np.r_[distinct_idx, uplift.size - 1]

    num_trmnt = _cumsum(treatment)[thresholds]
    y_trmnt_sum = _cumsum(y_trmnt)[thresholds]
    num_all = thresholds + 1
    num_ctrl = num_all - num_trmnt
    y_ctrl_sum = _cumsum(y_ctrl)[thresholds]

    curve = y_trmnt_sum - y_ctrl_sum * np.divide(
        num_trmnt, num_ctrl, out=np.zeros_like(num_trmnt), where=num_ctrl != 0
    )

    if num_all.size == 0 or curve[0] != 0 or num_all[0] != 0:
        num_all = np.r_[0, num_all]
        curve = np.r_[0, curve]

    return num_all, curve


def perfect_uplift_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    cr = np.sum((y_true == 1) & (treatment == 0))
    tn = np.sum((y_true == 0) & (treatment == 1))
    summand = y_true if cr > tn else treatment
    perfect_score = 2 * (y_true == treatment) + summand
    return uplift_curve(y_true, perfect_score, treatment)


def perfect_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    *,
    negative_effect: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    treatment = np.asarray(treatment)
    n = len(y_true)

    if negative_effect:
        score = y_true * treatment - y_true * (1 - treatment)
        return qini_curve(y_true, score, treatment)

    ratio = (
        y_true[treatment == 1].sum()
        - len(y_true[treatment == 1]) * y_true[treatment == 0].sum() / len(y_true[treatment == 0])
    )
    return np.array([0, ratio, n]), np.array([0, ratio, ratio])


def uplift_auc_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
) -> float:
    """Нормализованный AUUC ∈ [0, 1]."""
    x_act, y_act = uplift_curve(y_true, uplift, treatment)
    x_perf, y_perf = perfect_uplift_curve(y_true, treatment)
    x_base = np.array([0.0, x_perf[-1]])
    y_base = np.array([0.0, y_perf[-1]])

    base_auc = auc(x_base, y_base)
    perf_auc = auc(x_perf, y_perf) - base_auc
    act_auc = auc(x_act, y_act) - base_auc
    if perf_auc <= 0:
        return 0.0
    return float(np.clip(act_auc / perf_auc, 0.0, 1.0))


def qini_auc_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    *,
    negative_effect: bool = True,
) -> float:
    """Нормализованный Qini coefficient ∈ [0, 1]."""
    x_act, y_act = qini_curve(y_true, uplift, treatment)
    x_perf, y_perf = perfect_qini_curve(y_true, treatment, negative_effect=negative_effect)
    x_base = np.array([0.0, x_perf[-1]])
    y_base = np.array([0.0, y_perf[-1]])

    base_auc = auc(x_base, y_base)
    perf_auc = auc(x_perf, y_perf) - base_auc
    act_auc = auc(x_act, y_act) - base_auc
    if perf_auc <= 0:
        return 0.0
    return float(np.clip(act_auc / perf_auc, 0.0, 1.0))


def uplift_at_k(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float,
) -> float:
    """Абсолютный прирост конверсии в топ-k% по предсказанному uplift."""
    n = len(uplift)
    n_top = max(1, int(n * k))
    idx = np.argsort(uplift, kind="mergesort")[::-1][:n_top]
    y_top = y_true[idx]
    t_top = treatment[idx]
    tr = t_top == 1
    ct = ~tr
    if tr.sum() == 0 or ct.sum() == 0:
        return 0.0
    return float(y_top[tr].mean() - y_top[ct].mean())


def average_uplift(y_true: np.ndarray, treatment: np.ndarray) -> float:
    return float(y_true[treatment == 1].mean() - y_true[treatment == 0].mean())


def irr_score(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k: float,
) -> float:
    """IRR: во сколько раз uplift@k выше среднего по выборке."""
    base = average_uplift(y_true, treatment)
    if abs(base) < 1e-12:
        return 0.0
    return uplift_at_k(y_true, uplift, treatment, k) / base


def campaign_profitability(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    *,
    k: float,
    margin_per_conversion: float,
    treatment_cost: float,
) -> dict[str, float]:
    n_top = max(1, int(len(uplift) * k))
    idx = np.argsort(uplift, kind="mergesort")[::-1][:n_top]
    y_top = y_true[idx]
    t_top = treatment[idx]
    tr, ct = t_top == 1, t_top == 0

    inc_conv = 0.0
    if tr.sum() and ct.sum():
        inc_conv = float(y_top[tr].mean() - y_top[ct].mean())

    revenue = inc_conv * n_top * margin_per_conversion
    cost = n_top * treatment_cost
    profit = revenue - cost

    return {
        "k": k,
        "targeted_clients": float(n_top),
        "incremental_conversion_rate": inc_conv,
        "incremental_revenue": revenue,
        "campaign_cost": cost,
        "net_profit": profit,
        "roi": profit / cost if cost > 0 else 0.0,
    }


def evaluate_all_metrics(
    y_true: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    *,
    k: float,
    margin: float,
    cost: float,
    k_grid: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Полный набор метрик для одного предсказания."""
    result = {
        "auuc": uplift_auc_score(y_true, uplift, treatment),
        "qini": qini_auc_score(y_true, uplift, treatment),
        f"uplift_at_{int(k * 100)}pct": uplift_at_k(y_true, uplift, treatment, k),
        f"irr_at_{int(k * 100)}pct": irr_score(y_true, uplift, treatment, k),
        "avg_uplift": average_uplift(y_true, treatment),
    }
    profit = campaign_profitability(
        y_true, uplift, treatment, k=k, margin_per_conversion=margin, treatment_cost=cost
    )
    result["net_profit"] = profit["net_profit"]
    result["roi"] = profit["roi"]

    if k_grid:
        for kk in k_grid:
            pct = int(kk * 100)
            result[f"uplift_at_{pct}pct"] = uplift_at_k(y_true, uplift, treatment, kk)
            result[f"irr_at_{pct}pct"] = irr_score(y_true, uplift, treatment, kk)

    return result
