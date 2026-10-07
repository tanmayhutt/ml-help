"""All HTTP routes."""
from __future__ import annotations

import io
import json
from typing import Any

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

from . import config, db, limits, profile, samples, storage
from .content.glossary import GLOSSARY
from .content.lessons import LESSONS
from .ml import catalog, codegen, export
from .ml.tasks import detect_task
from .runner import KINDS, get_runner


def guard(request: Request) -> None:
    limits.require_token(request)
    limits.check(request, "api", config.RATE_REQUESTS_PER_MIN)


router = APIRouter(prefix="/api", dependencies=[Depends(guard)])


@router.get("/health")
def health() -> dict:
    r = get_runner()
    return {"ok": True, "queue": r.pending(), "running": r.current, "limits": {
        "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024), "max_rows": config.MAX_ROWS, "max_cols": config.MAX_COLS,
        "job_timeout_sec": config.JOB_TIMEOUT_SEC, "leaderboard_rows": config.LEADERBOARD_ROWS, "train_rows": config.TRAIN_ROWS,
        "tune_rows": config.TUNE_ROWS, "tune_max_iter": config.TUNE_MAX_ITER, "retention_days": config.RETENTION_DAYS,
        "n_jobs": config.N_JOBS, "queue_max": config.QUEUE_MAX}, "disk": storage.disk_usage()}


@router.get("/catalog")
def get_catalog() -> dict:
    return catalog.public_catalog()


@router.get("/glossary")
def glossary() -> list:
    return GLOSSARY


@router.get("/lessons")
def lessons() -> list:
    return LESSONS


@router.get("/samples")
def list_samples() -> list:
    return [{"key": k, **v} for k, v in samples.SAMPLES.items()]


# ---- datasets ------------------------------------------------------------

def _dataset_row(con, dataset_id: str) -> dict:
    row = db.row_to_dict(con.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone(), ("profile",))
    if row is None:
        raise HTTPException(404, "Dataset not found.")
    return row


def _register(df: pd.DataFrame, name: str, sample: bool) -> dict:
    dataset_id = storage.new_id()
    size = storage.save_dataset(df, dataset_id)
    prof = profile.profile(df)
    with db.connect() as con:
        con.execute("INSERT INTO datasets (id, name, rows, cols, bytes, sample, created, profile) VALUES (?,?,?,?,?,?,?,?)",
                    (dataset_id, name[:120], len(df), len(df.columns), size, int(sample), db.now(), json.dumps(prof, default=str)))
    return {"id": dataset_id, "name": name, "rows": len(df), "cols": len(df.columns), "profile": prof}


@router.get("/datasets")
def list_datasets() -> list:
    with db.connect() as con:
        rows = con.execute("SELECT id, name, rows, cols, bytes, sample, created FROM datasets ORDER BY created DESC").fetchall()
    return [dict(r) for r in rows]


@router.post("/datasets/upload")
async def upload(request: Request, file: UploadFile = File(...)) -> dict:
    limits.check(request, "upload", config.RATE_UPLOADS_PER_MIN)
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File is larger than {config.MAX_UPLOAD_BYTES // (1024*1024)} MB.")
    try:
        df = storage.parse_upload(file.filename or "upload.csv", raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not parse the file: {type(e).__name__}")
    storage.cleanup()
    return _register(df, file.filename or "upload.csv", sample=False)


@router.post("/datasets/sample/{key}")
def load_sample(key: str) -> dict:
    if key not in samples.SAMPLES:
        raise HTTPException(404, "Unknown sample.")
    with db.connect() as con:
        existing = con.execute("SELECT id, name, rows, cols FROM datasets WHERE sample = 1 AND name = ?", (samples.SAMPLES[key]["name"],)).fetchone()
    if existing:
        return {**dict(existing), "existing": True}
    try:
        df = samples.load(key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _register(df, samples.SAMPLES[key]["name"], sample=True)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> dict:
    with db.connect() as con:
        return _dataset_row(con, dataset_id)


@router.delete("/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    with db.connect() as con:
        _dataset_row(con, dataset_id)
        con.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    storage.delete_dataset_files(dataset_id)
    return {"ok": True}


class TaskQuery(BaseModel):
    target: str | None = None


@router.post("/datasets/{dataset_id}/task")
def task_for(dataset_id: str, body: TaskQuery) -> dict:
    try:
        df = storage.load_dataset(dataset_id)
    except FileNotFoundError:
        raise HTTPException(404, "Dataset not found.")
    if body.target and body.target not in df.columns:
        raise HTTPException(400, "Target column not found.")
    return detect_task(df, body.target)


# ---- jobs ----------------------------------------------------------------

class JobIn(BaseModel):
    kind: str
    dataset_id: str
    params: dict[str, Any] = Field(default_factory=dict)


@router.post("/jobs")
def create_job(request: Request, body: JobIn) -> dict:
    limits.check(request, "job", config.RATE_JOBS_PER_MIN)
    if body.kind not in KINDS:
        raise HTTPException(400, "Unknown job kind.")
    with db.connect() as con:
        _dataset_row(con, body.dataset_id)
    if len(json.dumps(body.params)) > 20_000:
        raise HTTPException(400, "Parameters too large.")
    try:
        job_id = get_runner().submit(body.kind, body.dataset_id, body.params)
    except RuntimeError as e:
        raise HTTPException(429, str(e))
    return {"id": job_id, "status": "queued", "position": get_runner().pending()}


@router.get("/jobs")
def list_jobs(dataset_id: str | None = None, limit: int = 50) -> list:
    limit = max(1, min(limit, 200))
    with db.connect() as con:
        if dataset_id:
            rows = con.execute("SELECT id, kind, dataset_id, status, created, finished, error, params FROM jobs WHERE dataset_id = ? ORDER BY created DESC LIMIT ?", (dataset_id, limit)).fetchall()
        else:
            rows = con.execute("SELECT id, kind, dataset_id, status, created, finished, error, params FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
    return [db.row_to_dict(r, ("params",)) for r in rows]


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with db.connect() as con:
        row = db.row_to_dict(con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone(), ("params", "progress", "result"))
    if row is None:
        raise HTTPException(404, "Job not found.")
    if row["status"] == "queued":
        row["queue"] = get_runner().pending()
    return row


# ---- models --------------------------------------------------------------

def _model_row(con, model_id: str) -> dict:
    row = db.row_to_dict(con.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone(), ("features", "metrics"))
    if row is None:
        raise HTTPException(404, "Model not found.")
    return row


@router.get("/models")
def list_models(dataset_id: str | None = None) -> list:
    with db.connect() as con:
        q = "SELECT id, job_id, dataset_id, name, task, target, features, metrics, created FROM models"
        rows = con.execute(q + (" WHERE dataset_id = ?" if dataset_id else "") + " ORDER BY created DESC", (dataset_id,) if dataset_id else ()).fetchall()
    out = []
    for r in rows:
        d = db.row_to_dict(r, ("features", "metrics"))
        f = d.pop("features", {}) or {}
        d["features"] = {"features": f.get("features", []), "classes": f.get("classes")}
        out.append(d)
    return out


class PredictIn(BaseModel):
    rows: list[dict[str, Any]] = Field(max_length=500)


@router.post("/models/{model_id}/predict")
def predict(model_id: str, body: PredictIn) -> dict:
    with db.connect() as con:
        m = _model_row(con, model_id)
    path = storage.model_path(model_id)
    if not path.exists():
        raise HTTPException(410, "Model file expired.")
    pipe = joblib.load(path)
    feats = m["features"]["features"]
    df = pd.DataFrame(body.rows)
    for f in feats:
        if f not in df.columns:
            df[f] = None
    df = df[feats]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore") if df[c].dtype == object else df[c]
    try:
        pred = pipe.predict(df)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Prediction failed: {type(e).__name__}: {str(e)[:200]}")
    out: dict = {"predictions": []}
    classes = m["features"].get("classes")
    if classes:
        out["predictions"] = [classes[int(p)] for p in pred]
        if hasattr(pipe, "predict_proba"):
            try:
                out["probabilities"] = [[round(float(x), 4) for x in r] for r in pipe.predict_proba(df)]
                out["classes"] = classes
            except Exception:
                pass
    else:
        out["predictions"] = [round(float(p), 6) for p in pred]
    return out


@router.post("/models/{model_id}/predict-file")
async def predict_file(model_id: str, file: UploadFile = File(...)) -> dict:
    raw = await file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large.")
    try:
        df = storage.parse_upload(file.filename or "rows.csv", raw)
    except ValueError as e:
        raise HTTPException(400, str(e))
    df = df.head(5000)
    rows = df.astype(object).where(df.notna(), None).to_dict(orient="records")
    res = predict(model_id, PredictIn(rows=rows[:500]))
    res["note"] = f"Predicted the first {min(len(rows), 500)} rows."
    return res


@router.get("/models/{model_id}/download")
def download_model(model_id: str) -> Response:
    with db.connect() as con:
        m = _model_row(con, model_id)
    path = storage.model_path(model_id)
    if not path.exists():
        raise HTTPException(410, "Model file expired.")
    return Response(path.read_bytes(), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="mlhelp-{model_id}.joblib"'})


@router.get("/models/{model_id}/notebook")
def download_notebook(model_id: str) -> Response:
    with db.connect() as con:
        m = _model_row(con, model_id)
    f = m["features"]
    code, imports = codegen.estimator_code(m["task"], f.get("spec") or {})
    nb = export.notebook({"name": m["name"], "task": m["task"], "target": m["target"], "features": f["features"], "numeric": f.get("numeric", []),
                          "categorical": f.get("categorical", []), "prep_options": f.get("prep_options", {}), "estimator_code": code, "imports": imports, "test_size": f.get("test_size", 0.2)})
    return Response(nb, media_type="application/x-ipynb+json", headers={"Content-Disposition": f'attachment; filename="mlhelp-{model_id}.ipynb"'})


@router.get("/models/{model_id}/code", response_class=PlainTextResponse)
def model_code(model_id: str) -> str:
    with db.connect() as con:
        m = _model_row(con, model_id)
    code, imports = codegen.estimator_code(m["task"], m["features"].get("spec") or {})
    return imports + f"\nmodel = {code}\n"
