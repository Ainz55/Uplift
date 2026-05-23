# TARNet/DragonNet Uplift Submission (bs8192 GPU run)

## Описание

Самодостаточная папка с моделью **TARNet с DragonNet propensity head**, обученной на uplift-задаче MAGNIT TECH (case 2).

**Параметры обучения:**
- Архитектура: TARNet, 256 hidden, 2 hidden layers + 2 head layers, DragonNet propensity head
- Optimizer: AdamW, lr=3e-4, cosine schedule
- Batch size: **8192** (GPU)
- Val fraction: 0.15 (stratified by treatment × spend>0)
- Early stopping: MSE on val (patience=10)
- Single seed: random_state=42
- Feature set: "semantic" (215 features after preprocessing)
- Final val_uplift@10: **22.65**
- LB result: **17.18**

## Структура

```
submission_bs8192/
├── neural_uplift.py          # Main training/inference script
├── data.py                   # Data loading & validation
├── features.py               # Feature engineering (semantic mode)
├── config.py                 # Config dataclass
├── runtime_env.py            # Runtime environment setup
├── requirements.txt          # Python dependencies
├── predictions_neural.csv    # Submission file (test predictions)
├── dataset/
│   ├── train.parquet         # Training data
│   └── test.parquet          # Test data
└── model/
    ├── tarnet.pt             # Model checkpoint (state_dict + scaler + feature_cols)
    ├── predictions.csv       # Same as predictions_neural.csv (rank-avg)
    └── predictions_neural_raw.csv  # Raw scale predictions (for ensembling)
```

## Воспроизведение

### Полное переобучение

```bash
python neural_uplift.py \
  --mode train \
  --dataset dataset \
  --output-dir output \
  --n-seeds 1 \
  --val-fraction 0.15 \
  --with-dragonnet \
  --strat-mode old \
  --epochs 80 \
  --patience 10 \
  --batch-size 8192
```

### Inference из сохранённого чекпойнта

```bash
python neural_uplift.py \
  --mode infer \
  --model-path model/tarnet.pt \
  --dataset dataset \
  --output-dir output
```

Выход: `output/predictions_neural.csv` — submission-ready CSV с колонками `user_id, UPLIFT_SCORE`.

## Зависимости

```
torch>=2.0
numpy
pandas
scikit-learn
lightgbm
pyarrow
```

См. `requirements.txt`.

## Метрика

Целевая метрика: **uplift@10 lower bound 80% CI** (bootstrap 200 итераций).

- val_uplift@10 (point): 22.65
- LB result: 17.18 (uplift_lower_80ci on public test)
