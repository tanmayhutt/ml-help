"""Dataset files on disk plus lazy retention cleanup (no cron, no background loop)."""
from __future__ import annotations

import io
import secrets
import shutil
import time
from pathlib import Path

import pandas as pd

from . import config, db

DATASETS = config.DATA_DIR / "datasets"
MODELS = config.DATA_DIR / "models"


def ensure_dirs() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)


def new_id() -> str:
    return secrets.token_hex(8)


def dataset_path(dataset_id: str) -> Path:
    return DATASETS / f"{dataset_id}.pkl"


def model_path(model_id: str) -> Path:
    return MODELS / f"{model_id}.joblib"


def parse_upload(filename: str, raw: bytes) -> pd.DataFrame:
    """Decode CSV, TSV, or Excel content. Signature is inspected, not just the extension."""
    name = filename.lower()
    if raw[:4] == b"PK\x03\x04" and (name.endswith(".xlsx") or name.endswith(".xlsm")):
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    elif name.endswith((".csv", ".tsv", ".txt")):
        sep = "\t" if name.endswith(".tsv") else None
        df = pd.read_csv(io.BytesIO(raw), sep=sep, engine="python", encoding_errors="replace")
    else:
        raise ValueError("Only .csv, .tsv, .txt and .xlsx files are accepted.")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The file has no rows or no columns.")
    if df.shape[0] > config.MAX_ROWS:
        raise ValueError(f"Too many rows ({df.shape[0]}). Limit is {config.MAX_ROWS}.")
    if df.shape[1] > config.MAX_COLS:
        raise ValueError(f"Too many columns ({df.shape[1]}). Limit is {config.MAX_COLS}.")
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    # Deduplicate column names
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            cols.append(c)
    df.columns = cols
    return classic_dtypes(df)


def classic_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Plain numpy dtypes only: object text with NaN for blanks, float/int numbers, bool. Nullable pandas
    extension dtypes (pd.NA) break scikit-learn's imputers, so they are never stored."""
    df = df.infer_objects()
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.StringDtype) or str(s.dtype) in ("boolean", "Int64", "Int32", "Float64", "Float32"):
            if str(s.dtype).startswith(("Int", "Float")):
                df[c] = s.astype("float64") if s.isna().any() else s.astype("int64" if str(s.dtype).startswith("Int") else "float64")
            elif str(s.dtype) == "boolean":
                df[c] = s.astype(object).where(s.notna(), None).astype(object)
            else:
                df[c] = s.astype(object).where(s.notna(), None)
        elif s.dtype == object:
            # text column: try numbers, otherwise keep text with None for blanks
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == s.notna().sum() and s.notna().any():
                df[c] = num
            else:
                df[c] = s.where(s.notna(), None)
                # strings only
                df[c] = df[c].map(lambda v: v if v is None else str(v))
    return df


def save_dataset(df: pd.DataFrame, dataset_id: str) -> int:
    ensure_dirs()
    p = dataset_path(dataset_id)
    df.to_pickle(p, protocol=5)
    return p.stat().st_size


def load_dataset(dataset_id: str) -> pd.DataFrame:
    p = dataset_path(dataset_id)
    if not p.exists():
        raise FileNotFoundError(dataset_id)
    return pd.read_pickle(p)


def delete_dataset_files(dataset_id: str) -> None:
    dataset_path(dataset_id).unlink(missing_ok=True)


def cleanup() -> dict:
    """Delete datasets, models and jobs older than the retention window. Called on upload, never on a timer."""
    cutoff = time.time() - config.RETENTION_DAYS * 86400
    removed = {"datasets": 0, "models": 0, "jobs": 0}
    with db.connect() as con:
        old = con.execute("SELECT id FROM datasets WHERE created < ? AND sample = 0", (cutoff,)).fetchall()
        for r in old:
            delete_dataset_files(r["id"])
            con.execute("DELETE FROM datasets WHERE id = ?", (r["id"],))
            removed["datasets"] += 1
        old_models = con.execute("SELECT id FROM models WHERE created < ?", (cutoff,)).fetchall()
        for r in old_models:
            model_path(r["id"]).unlink(missing_ok=True)
            con.execute("DELETE FROM models WHERE id = ?", (r["id"],))
            removed["models"] += 1
        # Keep job table bounded
        cur = con.execute(
            "DELETE FROM jobs WHERE id IN (SELECT id FROM jobs ORDER BY created DESC LIMIT -1 OFFSET ?)",
            (config.MAX_JOBS_KEPT,),
        )
        removed["jobs"] = cur.rowcount
        # Cap total user datasets
        extra = con.execute(
            "SELECT id FROM datasets WHERE sample = 0 ORDER BY created DESC LIMIT -1 OFFSET ?",
            (config.MAX_DATASETS,),
        ).fetchall()
        for r in extra:
            delete_dataset_files(r["id"])
            con.execute("DELETE FROM datasets WHERE id = ?", (r["id"],))
            removed["datasets"] += 1
    return removed


def disk_usage() -> dict:
    ensure_dirs()
    total = 0
    for p in list(DATASETS.glob("*")) + list(MODELS.glob("*")):
        total += p.stat().st_size
    free = shutil.disk_usage(config.DATA_DIR).free
    return {"used_bytes": total, "free_bytes": free}
