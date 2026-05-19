"""Конфигурация пайплайна uplift-моделирования."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "output"

RANDOM_STATE = 42
N_CV_FOLDS = 5
TOP_K_DEFAULT = 0.30
# Доля train-части фолда, отводимая под early stopping (стратификация по treatment_flg)
EARLY_STOPPING_VAL_FRACTION = 0.20

# Бизнес-параметры кампании (руб.)
# Средний инкрементальный чек и стоимость бонуса (настраиваются под бизнес)
DEFAULT_MARGIN = 1200.0
DEFAULT_TREATMENT_COST = 35.0

# Доли клиентов для анализа uplift@k и графиков
UPLIFT_K_GRID: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50)

# Валидный диапазон возраста в справочнике (есть выбросы: -7491, 1901)
AGE_MIN = 18
AGE_MAX = 90


@dataclass
class LGBMParams:
    n_estimators: int = 2000
    learning_rate: float = 0.03
    max_depth: int = 7
    num_leaves: int = 63
    min_child_samples: int = 40
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 80


@dataclass
class PipelineConfig:
    dataset_dir: Path = DATASET_DIR
    reports_dir: Path = REPORTS_DIR
    output_dir: Path = OUTPUT_DIR
    submission_path: Path = field(default_factory=lambda: PROJECT_ROOT / "submission.csv")
    random_state: int = RANDOM_STATE
    n_folds: int = N_CV_FOLDS
    top_k: float = TOP_K_DEFAULT
    margin: float = DEFAULT_MARGIN
    treatment_cost: float = DEFAULT_TREATMENT_COST
    lgbm: LGBMParams = field(default_factory=LGBMParams)
    uplift_k_grid: tuple[float, ...] = UPLIFT_K_GRID
    early_stopping_val_fraction: float = EARLY_STOPPING_VAL_FRACTION

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
