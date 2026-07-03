"""Central configuration: paths, seeds, model params, and column lists."""
from pathlib import Path

# Project root = two folders up from this file (src/nfl_draft/config.py).
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"

# The three raw competition files.
TRAIN_CSV  = DATA_RAW / "train.csv"
TEST_CSV   = DATA_RAW / "test.csv"
SAMPLE_SUB = DATA_RAW / "sample_submission.csv"

# CV / reproducibility knobs.
SEED    = 42
SEEDS   = [42, 123, 2025]   # seed-averaging set (steadies scores on the tiny 696-row test set)
N_FOLDS = 5

# Name of the column we're predicting.
TARGET = "Drafted"

# Age + the six combine drills; a MISSING value in any of these is itself a strong signal.
DRILL_COLS = ["Age", "Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
              "Broad_Jump", "Agility_3cone", "Shuttle"]

# The six measured drills only (no Age) — used for position-relative z-scores.
DRILL_VALS = ["Sprint_40yd", "Vertical_Jump", "Bench_Press_Reps",
              "Broad_Jump", "Agility_3cone", "Shuttle"]

# Categoricals we number-encode for the LightGBM model.
CAT_COLS = ["Player_Type", "Position_Type", "Position"]

# Categoricals CatBoost handles natively as raw strings (School stays a string here).
CAT_FEATS = ["School", "Player_Type", "Position_Type", "Position"]

# Columns excluded from the LightGBM feature matrix.
DROP_COLS = ["Id", "Drafted", "School"]

# Best LightGBM hyperparameters (found by Optuna in Phase 2).
LGBM_PARAMS = dict(
    n_estimators      = 1000,
    learning_rate     = 0.0645163786568917,
    num_leaves        = 41,
    max_depth         = 8,
    min_child_samples = 100,
    feature_fraction  = 0.5571356598017494,
    bagging_fraction  = 0.9242554434882404,
    bagging_freq      = 1,
    reg_alpha         = 0.0059161270802277665,
    reg_lambda        = 3.6116983011094534,
    verbose           = -1,
)

# CatBoost hyperparameters.
CATBOOST_PARAMS = dict(
    iterations            = 1000,
    learning_rate         = 0.05,
    depth                 = 6,
    eval_metric           = "AUC",
    early_stopping_rounds = 50,
    verbose               = 0,
)

# Ensemble blend weights (85% LightGBM / 15% CatBoost) and School-encoding smoothing.
W_LGBM = 0.85
W_CB   = 1 - W_LGBM
SMOOTH = 10
