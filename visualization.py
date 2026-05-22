"""Diagnostic plots for continuous uplift validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PipelineConfig
from evaluation import EvaluationReport
from metrics import average_uplift, qini_curve, uplift_at_k, uplift_curve

plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)


def plot_uplift_and_qini(
    y: np.ndarray,
    uplift: np.ndarray,
    treatment: np.ndarray,
    reports_dir: Path,
) -> None:
    n = max(len(y), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x_u, y_u = uplift_curve(y, uplift, treatment)
    axes[0].plot(x_u / n, y_u, label="model", color="#2563eb", lw=2)
    axes[0].axline((0, 0), slope=average_uplift(y, treatment) * n, color="#9ca3af", ls=":", label="random")
    axes[0].set_xlabel("targeted share")
    axes[0].set_ylabel("cumulative spend uplift")
    axes[0].set_title("Uplift curve")
    axes[0].legend()

    x_q, y_q = qini_curve(y, uplift, treatment)
    axes[1].plot(x_q / n, y_q, label="model", color="#7c3aed", lw=2)
    axes[1].axhline(0, color="#9ca3af", ls=":")
    axes[1].set_xlabel("targeted share")
    axes[1].set_ylabel("cumulative Qini")
    axes[1].set_title("Qini curve")
    axes[1].legend()

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
    labels = [f"{int(round(k * 100))}%" for k in k_grid]
    values = [uplift_at_k(y, uplift, treatment, k) for k in k_grid]
    avg = average_uplift(y, treatment)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color="#2563eb", edgecolor="white")
    ax.axhline(avg, color="#ef4444", ls="--", label=f"average={avg:.3f}")
    ax.set_xlabel("top K by predicted uplift")
    ax.set_ylabel("spend uplift")
    ax.set_title("Uplift@K")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "02_uplift_at_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cv_comparison(report: EvaluationReport, cfg: PipelineConfig, reports_dir: Path) -> None:
    pct = int(round(cfg.top_k * 100))
    key = f"uplift_at_{pct}pct_ci_lower"
    rows = []
    for name, cv in report.all_results.items():
        rows.append(
            {
                "model": name,
                f"uplift@{pct}": cv.summary.get(f"uplift_at_{pct}pct", 0.0),
                "lower80": cv.summary.get(key, 0.0),
                "auuc": cv.summary.get("auuc", 0.0),
            }
        )
    df = pd.DataFrame(rows).sort_values("lower80")

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.35)))
    ax.barh(df["model"], df["lower80"], color="#059669")
    ax.set_xlabel(f"lower 80% CI for uplift@{pct}%")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(reports_dir / "03_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(uplift: np.ndarray, reports_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(uplift, bins=80, color="#2563eb", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#ef4444", ls="--", lw=1.5, label="uplift = 0")
    ax.axvline(float(np.mean(uplift)), color="#16a34a", ls="--", lw=1.5, label=f"mean={np.mean(uplift):.3f}")
    ax.set_xlabel("predicted uplift score")
    ax.set_ylabel("clients")
    ax.set_title("OOF uplift score distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "04_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importances: pd.Series, reports_dir: Path, top_n: int = 20) -> None:
    top = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.index, top.values, color="#2563eb")
    ax.set_xlabel("LightGBM importance")
    ax.set_title(f"Top {top_n} features")
    fig.tight_layout()
    fig.savefig(reports_dir / "05_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(
    report: EvaluationReport,
    cfg: PipelineConfig,
    *,
    feature_importances: pd.Series | None = None,
) -> None:
    cfg.ensure_dirs()
    y, treatment, uplift = report.y, report.treatment, report.oof_uplift
    plot_uplift_and_qini(y, uplift, treatment, cfg.reports_dir)
    plot_uplift_at_k(y, uplift, treatment, cfg.uplift_k_grid, cfg.reports_dir)
    plot_cv_comparison(report, cfg, cfg.reports_dir)
    plot_score_distribution(uplift, cfg.reports_dir)
    if feature_importances is not None and len(feature_importances):
        plot_feature_importance(feature_importances, cfg.reports_dir)
