import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Competition Target Labels (12 Target Pathology Classes)
TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture"
]

NUM_CLASSES = len(TARGET_COLS)

# Data Parameters
IMAGE_SIZE = (256, 256)
NUM_SLICES = 32  # Standardized depth dimension for 3D MRI series
NUM_CHANNELS = 3 # Multi-planar or RGB conversion

# Model Parameters
BACKBONE_NAME = "convnext_small"
TEXT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
EMBED_DIM = 512
DROPOUT_RATE = 0.3

# Training Parameters
SEED = 42
N_FOLDS = 5
BATCH_SIZE = 4
NUM_WORKERS = 2
EPOCHS = 10
LR = 1e-4
WEIGHT_DECAY = 1e-2
AMP = True

# Inference Parameters
SUBMISSION_PATH = BASE_DIR / "submission.csv"
