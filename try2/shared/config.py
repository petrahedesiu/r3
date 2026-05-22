
import os
from datetime import datetime
from pathlib import Path

import torch

_THIS_FILE = os.path.abspath(__file__)
_SHARED_DIR = os.path.dirname(_THIS_FILE)
_TRY2_DIR = os.path.dirname(_SHARED_DIR)
_PROJECT_ROOT = os.path.dirname(_TRY2_DIR)


class ExperimentConfig:

    EXPERIMENT_NAME: str = "baseline"
    DESCRIPTION: str = "Base experiment configuration"

    DATA_DIR: str = os.path.join(_PROJECT_ROOT, "CROP1")
    OUTPUT_BASE: str = os.path.join(_TRY2_DIR, "results")

    NUM_CLASSES: int = 3
    CLASS_NAMES: tuple = ("BG", "AEAL", "AEAR")

    IMG_SIZE: int = 384
    ENCODER_NAME: str = "efficientnet-b4"
    ENCODER_WEIGHTS: str = "imagenet"
    IN_CHANNELS: int = 1
    ATTENTION_TYPE: str = "scse"

    NUM_EPOCHS: int = 10
    BATCH_SIZE: int = 4
    LR: float = 5e-5
    WEIGHT_DECAY: float = 1e-4
    GRAD_CLIP_NORM: float = 1.0
    NUM_WORKERS: int = 0

    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    TVERSKY_ALPHA: float = 0.3
    TVERSKY_BETA: float = 0.7
    LOVASZ_WEIGHT: float = 0.3

    OVERSAMPLE_FACTOR: int = 3
    ROI_PADDING: int = 50

    SCHEDULER_T0: int = 10
    SCHEDULER_TMULT: int = 2
    SCHEDULER_ETA_MIN: float = 1e-7

    VAL_SPLIT: float = 0.2
    RANDOM_SEED: int = 42

    DEVICE: str = "mps" if (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ) else "cpu"

    @classmethod
    def make_output_dir(cls, tag: str = "") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = f"{ts}_{tag}" if tag else ts
        out_dir = Path(cls.OUTPUT_BASE) / cls.EXPERIMENT_NAME / dirname
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    @classmethod
    def summary(cls) -> str:
        lines = [
            "=" * 70,
            f"EXPERIMENT: {cls.EXPERIMENT_NAME}",
            f"  {cls.DESCRIPTION}",
            "=" * 70,
            f"  DATA_DIR          : {cls.DATA_DIR}",
            f"  OUTPUT_BASE       : {cls.OUTPUT_BASE}",
            f"  NUM_CLASSES       : {cls.NUM_CLASSES}  {cls.CLASS_NAMES}",
            f"  IMG_SIZE          : {cls.IMG_SIZE}",
            f"  ENCODER_NAME      : {cls.ENCODER_NAME}",
            f"  ATTENTION_TYPE    : {cls.ATTENTION_TYPE}",
            f"  NUM_EPOCHS        : {cls.NUM_EPOCHS}",
            f"  BATCH_SIZE        : {cls.BATCH_SIZE}",
            f"  LR                : {cls.LR}",
            f"  WEIGHT_DECAY      : {cls.WEIGHT_DECAY}",
            f"  OVERSAMPLE_FACTOR : {cls.OVERSAMPLE_FACTOR}",
            f"  FOCAL_ALPHA       : {cls.FOCAL_ALPHA}",
            f"  FOCAL_GAMMA       : {cls.FOCAL_GAMMA}",
            f"  TVERSKY_ALPHA     : {cls.TVERSKY_ALPHA}",
            f"  TVERSKY_BETA      : {cls.TVERSKY_BETA}",
            f"  LOVASZ_WEIGHT     : {cls.LOVASZ_WEIGHT}",
            f"  DEVICE            : {cls.DEVICE}",
            "=" * 70,
        ]
        return "\n".join(lines)

    @classmethod
    def to_dict(cls) -> dict:
        return {
            k: v
            for k, v in vars(cls).items()
            if not k.startswith("_") and not callable(v)
        }
