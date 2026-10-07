"""One job at a time, each in its own spawned process with a hard timeout and a memory cap.

Why a process and not a thread: a thread cannot be killed, and sklearn leaks nothing
when the whole process exits. The parent stays small and responsive; the child dies
after every job, so the container's memory returns to baseline.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from typing import Any

from . import config, db, storage

KINDS = {"leaderboard", "train", "tune", "curve", "cluster", "reduce", "anomaly"}


def _child(job_id: str, kind: str, dataset_id: str, params: dict, deadline: float, conn) -> None:
    # Memory and thread caps inside the job process only.
    try:
        import resource
        cap = config.JOB_MEMORY_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except Exception:
        pass
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(config.N_JOBS))
    import warnings
    warnings.filterwarnings("ignore", message="Loky-backed parallel loops")  # forests run single-core inside the sandbox
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import joblib

        from .ml import codegen, train, unsupervised

        df = storage.load_dataset(dataset_id)

        def progress(msg: str, frac: float) -> None:
            conn.send({"type": "progress", "message": msg, "fraction": round(float(frac), 3)})

        fn = {
            "leaderboard": train.leaderboard, "train": train.train, "tune": train.tune, "curve": train.curve,
            "cluster": unsupervised.cluster, "reduce": unsupervised.reduce, "anomaly": unsupervised.anomaly,
        }[kind]
        result = fn(df, params, progress, deadline)
        pipe = result.pop("_pipeline", None)
        try:
            result["code"] = codegen.script(kind, params, result)
        except Exception as e:  # noqa: BLE001
            result["code"] = f"# code generation failed: {type(e).__name__}: {e}"
        model_id = None
        if pipe is not None:
            model_id = storage.new_id()
            joblib.dump(pipe, storage.model_path(model_id), compress=3)
            result["model_id"] = model_id
        conn.send({"type": "done", "result": result, "model_id": model_id})
    except MemoryError:
        conn.send({"type": "error", "error": f"The job ran out of memory (limit {config.JOB_MEMORY_MB} MB). Use fewer rows or columns."})
    except Exception as e:  # noqa: BLE001
        conn.send({"type": "error", "error": f"{type(e).__name__}: {e}"[:600], "trace": traceback.format_exc()[-2000:]})
    finally:
        conn.close()


class Runner:
    def __init__(self) -> None:
        self.q: queue.Queue[str] = queue.Queue()
        self.ctx = mp.get_context("spawn")
        self.current: str | None = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")
        self._thread.start()

    # ---- public -------------------------------------------------------
    def pending(self) -> int:
        return self.q.qsize() + (1 if self.current else 0)

    def submit(self, kind: str, dataset_id: str | None, params: dict) -> str:
        if kind not in KINDS:
            raise ValueError("Unknown job kind.")
        if self.q.qsize() >= config.QUEUE_MAX:
            raise RuntimeError(f"The queue is full ({config.QUEUE_MAX} jobs waiting). Try again in a minute.")
        job_id = storage.new_id()
        with db.connect() as con:
            con.execute(
                "INSERT INTO jobs (id, kind, dataset_id, params, status, created) VALUES (?,?,?,?,?,?)",
                (job_id, kind, dataset_id, json.dumps(params), "queued", db.now()),
            )
        self.q.put(job_id)
        return job_id

    # ---- worker loop --------------------------------------------------
    def _loop(self) -> None:
        while True:
            job_id = self.q.get()
            try:
                self._run(job_id)
            except Exception as e:  # noqa: BLE001
                self._finish(job_id, "error", error=f"runner failure: {e}")
            finally:
                self.current = None

    def _run(self, job_id: str) -> None:
        with db.connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            return
        self.current = job_id
        params = json.loads(row["params"])
        deadline = time.time() + config.JOB_TIMEOUT_SEC
        with db.connect() as con:
            con.execute("UPDATE jobs SET status='running', started=?, progress=? WHERE id=?", (db.now(), json.dumps({"message": "Starting", "fraction": 0}), job_id))
        parent, child = self.ctx.Pipe(duplex=False)
        proc = self.ctx.Process(target=_child, args=(job_id, row["kind"], row["dataset_id"], params, deadline, child), daemon=True)
        proc.start()
        child.close()
        outcome: dict[str, Any] | None = None
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            if parent.poll(min(remaining, 1.0)):
                try:
                    msg = parent.recv()
                except EOFError:
                    break
                if msg["type"] == "progress":
                    with db.connect() as con:
                        con.execute("UPDATE jobs SET progress=? WHERE id=?", (json.dumps({"message": msg["message"], "fraction": msg["fraction"]}), job_id))
                else:
                    outcome = msg
                    break
            elif not proc.is_alive():
                break
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
        proc.join(1)
        parent.close()
        if outcome is None:
            if time.time() >= deadline:
                self._finish(job_id, "error", error=f"Stopped: the job exceeded the {config.JOB_TIMEOUT_SEC} s budget. Use fewer rows, fewer models, or a cheaper model.")
            else:
                self._finish(job_id, "error", error=f"The job process exited unexpectedly (exit code {proc.exitcode}). It probably hit the memory limit.")
            return
        if outcome["type"] == "error":
            self._finish(job_id, "error", error=outcome["error"])
            return
        result = outcome["result"]
        self._finish(job_id, "done", result=result)
        if outcome.get("model_id"):
            with db.connect() as con:
                con.execute(
                    "INSERT INTO models (id, job_id, dataset_id, name, task, target, features, metrics, created) VALUES (?,?,?,?,?,?,?,?,?)",
                    (outcome["model_id"], job_id, row["dataset_id"], result.get("model_name", "model"), result.get("task", ""), result.get("target"),
                     json.dumps({"features": result.get("features", []), "classes": result.get("classes"), "prep_options": result.get("preprocessing", {}).get("options", {}),
                                 "numeric": result.get("preprocessing", {}).get("columns", {}).get("numeric", []), "categorical": result.get("preprocessing", {}).get("columns", {}).get("categorical", []),
                                 "spec": params.get("spec") or {"kind": "single", "model": params.get("model")}, "test_size": result.get("test_size")}),
                     json.dumps(result.get("evaluation", {}).get("metrics", {})), db.now()),
                )

    def _finish(self, job_id: str, status: str, result: dict | None = None, error: str | None = None) -> None:
        with db.connect() as con:
            con.execute(
                "UPDATE jobs SET status=?, result=?, error=?, finished=?, progress=NULL WHERE id=?",
                (status, json.dumps(result, default=str) if result is not None else None, error, db.now(), job_id),
            )


runner: Runner | None = None


def get_runner() -> Runner:
    global runner
    if runner is None:
        runner = Runner()
    return runner
