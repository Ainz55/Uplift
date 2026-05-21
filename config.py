"""Configuration for the MAGNIT TECH uplift pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_FILE = "train.parquet"
TEST_FILE = "test.parquet"
PREDICTIONS_FILE = "predictions.csv"

RANDOM_STATE = 42
N_CV_FOLDS = 5
TOP_K_DEFAULT = 0.10
UPLIFT_K_GRID: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)


@dataclass
class LGBMParams:
    n_estimators: int = 900
    learning_rate: float = 0.035
    max_depth: int = 7
    num_leaves: int = 63
    min_child_samples: int = 80
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 80
    n_jobs: int = 4


@dataclass
class PipelineConfig:
    dataset_dir: Path = DATASET_DIR
    train_path: Path | None = None
    test_path: Path | None = None
    reports_dir: Path = REPORTS_DIR
    output_dir: Path = OUTPUT_DIR
    predictions_path: Path = field(default_factory=lambda: PROJECT_ROOT / PREDICTIONS_FILE)
    random_state: int = RANDOM_STATE
    n_folds: int = N_CV_FOLDS
    top_k: float = TOP_K_DEFAULT
    uplift_k_grid: tuple[float, ...] = UPLIFT_K_GRID
    early_stopping_val_fraction: float = 0.20
    bootstrap_iterations: int = 200
    bootstrap_ci: float = 0.80
    max_categories: int = 40
    lgbm: LGBMParams = field(default_factory=LGBMParams)

    def resolved_train_path(self) -> Path:
        return self.train_path or self.dataset_dir / TRAIN_FILE

    def resolved_test_path(self) -> Path:
        return self.test_path or self.dataset_dir / TEST_FILE

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path.parent.mkdir(parents=True, exist_ok=True)
