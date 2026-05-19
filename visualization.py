"""Визуализация результатов uplift-моделирования."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig
from evaluation import EvaluationReport
from metrics import (
    average_uplift,
    campaign_profitability,
    perfect_qini_curve,
    perfect_uplift_curve,
    qini_curve,
    uplift_at_k,
    uplift_curve,
)

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def _normalize_x(x: np.ndarray, n: int) -> np.ndarray:
    return x / n if n > 0 else x


def plot_uplift_and_qini(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    reports_dir: Path,
    *,
    title_suffix: str = "",
) -> None:
    n = len(y)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Uplift curve ---
    ax = axes[0]
    x_m, y_m = uplift_curve(y, uplift, treatment)
    x_p, y_p = perfect_uplift_curve(y, treatment)
    x_m_n = _normalize_x(x_m, n)
    x_p_n = _normalize_x(x_p, n)

    random_line = np.linspace(0, x_p_n[-1], 50)
    random_uplift = average_uplift(y, treatment) * random_line

    ax.plot(x_m_n, y_m, label="Модель (OOF)", color="#2563eb", lw=2)
    ax.plot(x_p_n, y_p, label="Идеальная", color="#16a34a", lw=1.5, ls="--")
    ax.plot(random_line, random_uplift, label="Случайный таргетинг", color="#9ca3af", lw=1.5, ls=":")
    ax.set_xlabel("Доля охваченных клиентов")
    ax.set_ylabel("Накопленный uplift")
    ax.set_title(f"Кривая Uplift{title_suffix}")
    ax.legend(loc="lower right")

    # --- Qini curve ---
    ax = axes[1]
    x_q, y_q = qini_curve(y, uplift, treatment)
    x_qp, y_qp = perfect_qini_curve(y, treatment)
    x_q_n = _normalize_x(x_q, n)
    x_qp_n = _normalize_x(x_qp, n)

    ax.plot(x_q_n, y_q, label="Модель (OOF)", color="#7c3aed", lw=2)
    ax.plot(x_qp_n, y_qp, label="Идеальная", color="#16a34a", lw=1.5, ls="--")
    ax.plot([0, 1], [0, y_qp[-1] / n if n else 0], label="Случайная", color="#9ca3af", lw=1.5, ls=":")
    ax.set_xlabel("Доля охваченных клиентов")
    ax.set_ylabel("Накопленный Qini")
    ax.set_title(f"Кривая Qini{title_suffix}")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(reports_dir / "01_uplift_qini_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uplift_at_k(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k_grid: tuple[float, ...],
    reports_dir: Path,
) -> None:
    avg = average_uplift(y, treatment)
    ks = list(k_grid)
    uplifts = [uplift_at_k(y, uplift, treatment, k) for k in ks]
    irrs = [u / avg if avg else 0 for u in uplifts]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{int(k*100)}%" for k in ks]

    axes[0].bar(labels, uplifts, color="#2563eb", edgecolor="white")
    axes[0].axhline(avg, color="#ef4444", ls="--", label=f"Средний uplift={avg:.3f}")
    axes[0].set_ylabel("Абсолютный прирост конверсии")
    axes[0].set_title("Uplift@top-K%")
    axes[0].legend()

    axes[1].bar(labels, irrs, color="#7c3aed", edgecolor="white")
    axes[1].axhline(1.0, color="#ef4444", ls="--", label="IRR = 1 (нет выигрыша)")
    axes[1].set_ylabel("IRR (отношение к среднему)")
    axes[1].set_title("IRR@top-K%")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(reports_dir / "02_uplift_irr_by_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(report: EvaluationReport, reports_dir: Path) -> None:
    rows = []
    for name, cv in report.all_results.items():
        rows.append({"model": name, **{k: v for k, v in cv.summary.items() if k != "weights"}})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models = df["model"]
    axes[0].barh(models, df["auuc"], color="#2563eb")
    axes[0].set_xlabel("AUUC (нормализованный, 0–1)")
    axes[0].set_title("Сравнение моделей: AUUC")
    axes[0].set_xlim(0, max(df["auuc"].max() * 1.15, 0.05))

    axes[1].barh(models, df["qini"], color="#7c3aed")
    axes[1].set_xlabel("Qini (нормализованный, 0–1)")
    axes[1].set_title("Сравнение моделей: Qini")
    axes[1].set_xlim(0, max(df["qini"].max() * 1.15, 0.05))

    fig.tight_layout()
    fig.savefig(reports_dir / "03_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Детализация по фолдам лучшей не-ансамблевой модели
    best = report.best_model_name
    if best != "ensemble" and best in report.all_results:
        fold_df = report.all_results[best].fold_metrics
        fig, ax = plt.subplots(figsize=(8, 4))
        x = fold_df["fold"]
        ax.plot(x, fold_df["auuc"], "o-", label="AUUC", color="#2563eb")
        ax.plot(x, fold_df["qini"], "s-", label="Qini", color="#7c3aed")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Score")
        ax.set_title(f"Стабильность CV: {best}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(reports_dir / "04_cv_folds.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_profitability(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    k_grid: tuple[float, ...],
    margin: float,
    cost: float,
    reports_dir: Path,
) -> None:
    profits, rois, ks = [], [], []
    for k in k_grid:
        r = campaign_profitability(
            y, uplift, treatment, k=k,
            margin_per_conversion=margin, treatment_cost=cost,
        )
        profits.append(r["net_profit"] / 1e6)
        rois.append(r["roi"] * 100)
        ks.append(k)

    labels = [f"{int(k*100)}%" for k in ks]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(labels, profits, color="#059669")
    axes[0].set_ylabel("Чистая прибыль, млн руб.")
    axes[0].set_title("Рентабельность кампании по доле охвата")
    axes[0].axhline(0, color="black", lw=0.8)

    axes[1].bar(labels, rois, color="#d97706")
    axes[1].set_ylabel("ROI, %")
    axes[1].set_title("ROI по доле охвата")
    axes[1].axhline(0, color="black", lw=0.8)

    fig.tight_layout()
    fig.savefig(reports_dir / "05_profitability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(uplift: np.ndarray, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(uplift, bins=80, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#ef4444", ls="--", lw=1.5, label="uplift = 0")
    ax.axvline(np.mean(uplift), color="#16a34a", ls="--", lw=1.5, label=f"mean={np.mean(uplift):.3f}")
    ax.set_xlabel("Предсказанный uplift")
    ax.set_ylabel("Число клиентов")
    ax.set_title("Распределение OOF-предсказаний uplift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "06_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, reports_dir: Path, top_n: int = 15) -> None:
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="#2563eb")
    ax.set_xlabel("Важность (LightGBM)")
    ax.set_title(f"Топ-{top_n} признаков")
    fig.tight_layout()
    fig.savefig(reports_dir / "07_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    report: EvaluationReport,
    cfg: PipelineConfig,
    *,
    feature_importances: pd.Series | None = None,
) -> None:
    cfg.ensure_dirs()
    y, t, uplift = report.y, report.treatment, report.oof_uplift

    plot_uplift_and_qini(y, uplift, t, cfg.reports_dir, title_suffix=" (OOF)")
    plot_uplift_at_k(y, uplift, t, cfg.uplift_k_grid, cfg.reports_dir)
    plot_cv_comparison(report, cfg.reports_dir)
    plot_profitability(y, uplift, t, cfg.uplift_k_grid, cfg.margin, cfg.treatment_cost, cfg.reports_dir)
    plot_score_distribution(uplift, cfg.reports_dir)

    if feature_importances is not None and len(feature_importances):
        plot_feature_importance(feature_importances, cfg.reports_dir)

    logger_msg = f"Графики сохранены в {cfg.reports_dir}"
    return logger_msg
