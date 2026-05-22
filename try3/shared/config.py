
import os
from pathlib import Path
import torch

_THIS_FILE = os.path.abspath(__file__)
_SHARED_DIR = os.path.dirname(_THIS_FILE)
_TRY3_DIR = os.path.dirname(_SHARED_DIR)
PROJECT_ROOT = os.path.dirname(_TRY3_DIR)

DATA_DIRS = [
    os.path.join(PROJECT_ROOT, "CROP1"),
    os.path.join(PROJECT_ROOT, 'CROP - februarie 2026'),
]

OUTPUT_BASE = os.path.join(_TRY3_DIR, "results")

NUM_CLASSES = 3
CLASS_NAMES = ("BG", "AEAL", "AEAR")

NUM_EPOCHS = 30
BATCH_SIZE = 4
LR = 5e-5
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
TVERSKY_ALPHA = 0.2
TVERSKY_BETA = 0.8
LOVASZ_WEIGHT = 0.3

SCHEDULER_T0 = 10
SCHEDULER_TMULT = 2
SCHEDULER_ETA_MIN = 1e-7

VAL_SPLIT = 0.2
RANDOM_SEED = 42

# pick the device
DEVICE = "mps" if (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
) else "cpu"
