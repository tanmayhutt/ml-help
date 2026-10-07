"""Every resource cap lives here. All values can be overridden by environment variables.

The defaults are tuned for a shared 4-core server: the app may never use more than
two cores, one training job runs at a time, and every job has a hard wall-clock
and memory limit enforced in a separate process.
"""
from __future__ import annotations

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


DATA_DIR = Path(os.environ.get("ML_DATA_DIR", "./data")).resolve()
STATIC_DIR = Path(os.environ.get("ML_STATIC_DIR", "../frontend")).resolve()

# Optional shared access token. When set, every /api call must send X-Access-Token.
ACCESS_TOKEN = os.environ.get("ML_ACCESS_TOKEN", "").strip()

# Upload limits
MAX_UPLOAD_BYTES = _int("ML_MAX_UPLOAD_MB", 20) * 1024 * 1024
MAX_ROWS = _int("ML_MAX_ROWS", 200_000)
MAX_COLS = _int("ML_MAX_COLS", 200)
MAX_DATASETS = _int("ML_MAX_DATASETS", 40)
RETENTION_DAYS = _int("ML_RETENTION_DAYS", 7)

# Compute limits
N_JOBS = _int("ML_N_JOBS", 2)                      # cores sklearn may use
QUEUE_MAX = _int("ML_QUEUE_MAX", 6)                 # pending jobs allowed
JOB_TIMEOUT_SEC = _int("ML_JOB_TIMEOUT_SEC", 180)   # hard kill per job
JOB_MEMORY_MB = _int("ML_JOB_MEMORY_MB", 2048)      # RLIMIT_AS for the job process
LEADERBOARD_ROWS = _int("ML_LEADERBOARD_ROWS", 5000)  # subsample for the model race
TRAIN_ROWS = _int("ML_TRAIN_ROWS", 50_000)          # subsample for single-model training
TUNE_ROWS = _int("ML_TUNE_ROWS", 5000)
TUNE_MAX_ITER = _int("ML_TUNE_MAX_ITER", 20)
CV_FOLDS = _int("ML_CV_FOLDS", 3)
SLOW_MODEL_ROWS = _int("ML_SLOW_MODEL_ROWS", 3000)  # SVM/KNN/MLP skipped above this in the race
EMBED_ROWS = _int("ML_EMBED_ROWS", 1500)            # t-SNE cap
PERM_IMPORTANCE_ROWS = _int("ML_PERM_ROWS", 1500)
MAX_JOBS_KEPT = _int("ML_MAX_JOBS_KEPT", 200)

# Abuse limits (per client IP, sliding minute)
RATE_UPLOADS_PER_MIN = _int("ML_RATE_UPLOADS_PER_MIN", 6)
RATE_JOBS_PER_MIN = _int("ML_RATE_JOBS_PER_MIN", 10)
RATE_REQUESTS_PER_MIN = _int("ML_RATE_REQUESTS_PER_MIN", 600)

RANDOM_STATE = 42
