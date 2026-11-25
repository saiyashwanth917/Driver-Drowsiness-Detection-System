from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "dataset_validation"
    / "loso_splits"
    / "nthu_loso_manifest.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "training"
    / "nthu_loso"
)

# ============================================================
# DATA CONFIGURATION
# ============================================================

IMAGE_SIZE = (160, 160)
CHANNELS = 3

CLASS_NAMES = [
    "notdrowsy",
    "drowsy",
]

CLASS_TO_INDEX = {
    "notdrowsy": 0,
    "drowsy": 1,
}

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_SEED = 42

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 1e-3

EARLY_STOPPING_PATIENCE = 6

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "mobilenetv2_nthu_loso_v1"

DROPOUT_RATE = 0.30

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

NORMALIZATION = "rescale_0_1"

# ============================================================
# VALIDATION
# ============================================================

# Subject-level validation only.
# Never randomly split adjacent frames into train/validation.
VALIDATION_SUBJECT_COUNT = 1