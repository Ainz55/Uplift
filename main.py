"""MAGNIT TECH case 2: uplift modeling for rec_spend."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

from runtime_env import configure_runtime_storage

configure_runtime_storage()

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

from config import PipelineConfig
from console_report import print_full_report
from data import ID_COL, TARGET_COL, TREATMENT_COL, DatasetError, load_and_validate
from evaluation import select_best_model, train_final_model
from features import prepare_datasets
from metrics import evaluate_all_metrics
from visualization import generate_all_plots

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("uplift")


def save_reports(
    cfg: PipelineConfig,
    eval_report,
    oof_metrics: dict[str, float],
    cv_tables: dict[str, pd.DataFrame],
) -> None:
    cfg.ensure_dirs()
    pct = int(round(cfg.top_k * 100))
    leaderboard_key = f"uplift_at_{pct}pct_ci_lower"
    summary = {
        "best_model": eval_report.best_model_name,
        "selection_metric": leaderboard_key,
        "leaderboard_proxy": float(eval_report.best_cv.summary[leaderboard_key]),
        "target_score": 20.00123,
        "gap_vs_target": float(eval_report.best_cv.summary[leaderboard_key] - 20.00123),
        "feature_set": cfg.feature_set,
        "oof_metrics": oof_metrics,
        "models_cv_summary": {
            name: {k: v for k, v in cv.summary.items() if k != "weights"}
            for name, cv in eval_report.all_results.items()
        },
    }
    if eval_report.ensemble_weights:
        summary["ensemble_weights"] = eval_report.ensemble_weights

    with open(cfg.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, fold_df in cv_tables.items():
        fold_df.to_csv(cfg.output_dir / f"cv_{name}.csv", index=False)

    logger.info("Reports saved to %s", cfg.output_dir)


def run(cfg: PipelineConfig) -> None:
    cfg.ensure_dirs()
    logger.info("Loading train=%s", cfg.resolved_train_path())
    logger.info("Loading test=%s", cfg.resolved_test_path())

    try:
        train, test, info = load_and_validate(
            cfg.dataset_dir,
            train_path=cfg.train_path,
            test_path=cfg.test_path,
        )
    except DatasetError as exc:
        logger.error("Dataset error: %s", exc)
        raise SystemExit(1) from exc

    X, X_test, feature_cols, _ = prepare_datasets(
        train,
        test,
        max_categories=cfg.max_categories,
        feature_set=cfg.feature_set,
    )
    y = train[TARGET_COL].to_numpy(dtype=np.float32)
    treatment = train[TREATMENT_COL].to_numpy(dtype=np.int8)

    logger.info("Cross-validation and model selection...")
    eval_report = select_best_model(X, y, treatment, cfg)
    oof_metrics = evaluate_all_metrics(
        y,
        eval_report.oof_uplift,
        treatment,
        k=cfg.top_k,
        k_grid=cfg.uplift_k_grid,
        bootstrap_iterations=cfg.bootstrap_iterations,
        bootstrap_ci=cfg.bootstrap_ci,
        random_state=cfg.random_state,
    )

    cv_tables = {name: cv.fold_metrics for name, cv in eval_report.all_results.items()}
    save_reports(cfg, eval_report, oof_metrics, cv_tables)

    logger.info("Training final model: %s", eval_report.best_model_name)
    weights = (
        eval_report.ensemble_weights
        if eval_report.best_model_name in {"ensemble", "rank_ensemble", "top_ensemble", "rank_top_ensemble"}
        else None
    )
    final_model = train_final_model(
        eval_report.best_model_name,
        X,
        y,
        treatment,
        cfg,
        sub_weights=weights,
    )

    feature_importances = None
    if hasattr(final_model, "feature_importances_"):
        feature_importances = final_model.feature_importances_(feature_cols)
    if (feature_importances is None or not len(feature_importances)) and hasattr(final_model, "models"):
        for sub_model in final_model.models:
            if hasattr(sub_model, "feature_importances_"):
                feature_importances = sub_model.feature_importances_(feature_cols)
                break

    logger.info("Generating validation plots...")
    generate_all_plots(eval_report, cfg, feature_importances=feature_importances)

    test_uplift = np.asarray(final_model.predict(X_test), dtype=float)
    test_uplift = np.nan_to_num(test_uplift, nan=0.0, posinf=0.0, neginf=0.0)
    predictions = pd.DataFrame(
        {
            ID_COL: test[ID_COL].to_numpy(),
            "UPLIFT_SCORE": test_uplift,
        }
    )
    predictions.to_csv(cfg.predictions_path, index=False, encoding="utf-8")

    print_full_report(
        info=info,
        n_features=len(feature_cols),
        cfg=cfg,
        report=eval_report,
        oof_metrics=oof_metrics,
        test_uplift=test_uplift,
        predictions_path=str(cfg.predictions_path),
    )
    logger.info("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuous uplift modeling for MAGNIT TECH case 2.",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Directory with train.parquet and test.parquet")
    parser.add_argument("--train", type=Path, default=None, help="Explicit train file path")
    parser.add_argument("--test", type=Path, default=None, help="Explicit test file path")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None, help="Output predictions.csv path")
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--top-k", type=float, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument(
        "--feature-set",
        choices=("baseline", "event_zero", "semantic", "enhanced"),
        default=None,
        help="baseline is the safer public-LB mode; event_zero only zero-fills event counters; semantic uses broader domain-aware imputations; enhanced adds experimental ratio/interactions",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names, e.g. hurdle_t_learner,t_learner_log",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    cfg = PipelineConfig()
    if args.dataset:
        cfg.dataset_dir = args.dataset
    if args.train:
        cfg.train_path = args.train
    if args.test:
        cfg.test_path = args.test
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.reports_dir:
        cfg.reports_dir = args.reports_dir
    if args.predictions:
        cfg.predictions_path = args.predictions
    elif args.output_dir:
        cfg.predictions_path = args.output_dir / "predictions.csv"
    if args.folds is not None:
        cfg.n_folds = args.folds
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.bootstrap_iterations is not None:
        cfg.bootstrap_iterations = args.bootstrap_iterations
    if args.max_categories is not None:
        cfg.max_categories = args.max_categories
    if args.feature_set is not None:
        cfg.feature_set = args.feature_set
    if args.models:
        cfg.model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())

    run(cfg)


if __name__ == "__main__":
    main()


# --- Run commands ---
# Best run v2 (new X-learner + hurdle variants, baseline features):
# python main.py --dataset dataset --output-dir output/v2 --predictions output/v2/predictions.csv --folds 5 --models hurdle_t_learner_s101,hurdle_t_learner_s201,x_learner,x_learner_s101,hurdle_x_learner,hurdle_x_learner_s101
#
# Best run v2 with enhanced features (adds ratio/interaction features):
# python main.py --dataset dataset --output-dir output/v2_enhanced --predictions output/v2_enhanced/predictions.csv --folds 5 --feature-set enhanced --models hurdle_t_learner_s101,x_learner,x_learner_s101,hurdle_x_learner,hurdle_x_learner_s101
#
# Quick probe (3 folds, 100 bootstrap, fast):
# python main.py --dataset dataset --output-dir output/probe --predictions output/probe/predictions.csv --folds 3 --bootstrap-iterations 100 --models x_learner,hurdle_x_learner,hurdle_t_learner_s101
