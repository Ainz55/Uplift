"""Basic EDA for the official MAGNIT uplift dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import COMMUNICATION_COL, TARGET_COL, TREATMENT_COL, load_and_validate


def _save_bar(series: pd.Series, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series.plot(kind="bar", ax=ax, color="#2563eb", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(dataset: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, info = load_and_validate(dataset)

    summary = pd.DataFrame(
        [
            {"metric": "train_rows", "value": info.n_train},
            {"metric": "test_rows", "value": info.n_test},
            {"metric": "raw_feature_columns", "value": info.n_features_raw},
            {"metric": "treatment_rate", "value": info.treatment_rate},
            {"metric": "rec_spend_mean", "value": info.spend_mean},
            {"metric": "rec_spend_zero_rate", "value": info.spend_zero_rate},
            {"metric": "control_mean_spend", "value": info.spend_control_mean},
            {"metric": "treatment_mean_spend", "value": info.spend_treatment_mean},
            {"metric": "average_uplift", "value": info.avg_uplift},
        ]
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    treatment = train[TREATMENT_COL]
    spend = train[TARGET_COL]

    spend.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999]).to_csv(
        output_dir / "rec_spend_describe.csv"
    )

    by_treatment = train.groupby(TREATMENT_COL)[TARGET_COL].agg(
        count="count",
        mean="mean",
        std="std",
        median="median",
        zero_rate=lambda s: float((s == 0).mean()),
        p90=lambda s: float(s.quantile(0.9)),
        p99=lambda s: float(s.quantile(0.99)),
        max="max",
    )
    by_treatment.to_csv(output_dir / "rec_spend_by_treatment.csv")

    by_comm = train.pivot_table(
        index=COMMUNICATION_COL,
        columns=TREATMENT_COL,
        values=TARGET_COL,
        aggfunc=["count", "mean", "median"],
    )
    by_comm.to_csv(output_dir / "rec_spend_by_communication.csv")

    missing = pd.DataFrame(
        {
            "train_missing_rate": train.isna().mean(),
            "test_missing_rate": test.isna().mean(),
        }
    ).sort_values("train_missing_rate", ascending=False)
    missing.to_csv(output_dir / "missing_values.csv")

    numeric_cols = [
        c
        for c in train.columns
        if c not in {TARGET_COL, TREATMENT_COL}
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    corr = (
        train[numeric_cols + [TARGET_COL]]
        .corr(numeric_only=True)[TARGET_COL]
        .drop(index=TARGET_COL, errors="ignore")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(30)
    )
    corr.to_csv(output_dir / "top_correlations_with_rec_spend.csv")

    _save_bar(
        treatment.value_counts().sort_index(),
        output_dir / "01_treatment_balance.png",
        "Treatment balance",
        "users",
    )
    _save_bar(
        train[COMMUNICATION_COL].value_counts().sort_index(),
        output_dir / "02_communication_types.png",
        "Communication type distribution",
        "users",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positive = spend[spend > 0]
    ax.hist(np.log1p(positive), bins=80, color="#059669", edgecolor="white")
    ax.set_title("Positive rec_spend distribution, log1p scale")
    ax.set_xlabel("log1p(rec_spend)")
    ax.set_ylabel("users")
    fig.tight_layout()
    fig.savefig(output_dir / "03_positive_rec_spend_log_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    missing.head(25)["train_missing_rate"].iloc[::-1].plot(kind="barh", ax=ax, color="#7c3aed")
    ax.set_title("Top missing feature rates")
    ax.set_xlabel("missing rate")
    fig.tight_layout()
    fig.savefig(output_dir / "04_missing_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"))
    args = parser.parse_args()
    run_eda(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
