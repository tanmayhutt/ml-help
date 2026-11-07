"""Checks for the newer features against a running server, then runs every generated script.

    ML_DATA_DIR=<same dir as the server> python tests/features.py http://127.0.0.1:8000

Covers: validation curve, explain (partial dependence + one row), feature selection (4 methods),
calibration, grid search, XGBoost, LightGBM, polynomial features, binning, and resampling
(oversample, undersample, SMOTE). Each generated script must run without errors.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(method, path, body=None, raw=False):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            out = r.read()
            return out if raw else json.loads(out)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read()[:300]}")


def run(kind, did, params):
    job = call("POST", "/api/jobs", {"kind": kind, "dataset_id": did, "params": params})
    t0 = time.time()
    while True:
        j = call("GET", f"/api/jobs/{job['id']}")
        if j["status"] in ("done", "error"):
            return j, time.time() - t0
        time.sleep(0.7)


from app import storage  # noqa: E402

work = tempfile.mkdtemp(prefix="mlhelp-features-")
fails = []


def check(label, kind, did, params, expect=None):
    j, secs = run(kind, did, params)
    if j["status"] != "done":
        fails.append(label)
        print(f"FAIL {label:42} {j.get('error')}")
        return None
    r = j["result"]
    problem = expect(r) if expect else None
    code = r.get("code", "")
    script_ok = "skipped"
    if code and not code.startswith("# no code"):
        path = os.path.join(work, f"{label.replace(' ', '_').replace('/', '-')}.py")
        open(path, "w").write(code)
        p = subprocess.run([sys.executable, path], cwd=work, capture_output=True, text=True, timeout=240)
        script_ok = "script ok" if p.returncode == 0 else "SCRIPT FAILED"
        if p.returncode != 0:
            fails.append(label + " (script)")
            print(p.stderr[-800:])
    if problem:
        fails.append(label)
    print(f"{'FAIL' if problem else 'ok  '} {label:42} {secs:5.1f}s  {script_ok}  {problem or ''}")
    return r


bc = call("POST", "/api/datasets/sample/breast_cancer")["id"]
dia = call("POST", "/api/datasets/sample/diabetes")["id"]
wine = call("POST", "/api/datasets/sample/wine")["id"]
C = {"task": "classification", "target": "diagnosis"}
R = {"task": "regression", "target": "progression"}
W = {"task": "classification", "target": "cultivar"}


def csv_for(did):
    storage.load_dataset(did).to_csv(os.path.join(work, "your_dataset.csv"), index=False)


# Classification on breast cancer
csv_for(bc)
check("xgboost train", "train", bc, C | {"spec": {"kind": "single", "model": "xgb"}}, lambda r: None if r["evaluation"]["metrics"]["accuracy"] > 0.9 else "low accuracy")
tr = check("lightgbm train", "train", bc, C | {"spec": {"kind": "single", "model": "lgbm"}}, lambda r: None if r["evaluation"]["metrics"]["accuracy"] > 0.9 else "low accuracy")
check("race includes xgb and lgbm", "leaderboard", bc, C, lambda r: None if {"xgb", "lgbm"} <= {x["key"] for x in r["leaderboard"] if x["status"] == "ok"} else "missing")
check("polynomial degree 2", "train", bc, C | {"spec": {"kind": "single", "model": "logreg"}, "preprocess": {"poly_degree": 2, "drop_columns": [c for c in storage.load_dataset(bc).columns if c.startswith("worst") or c.endswith("error")]}},
      lambda r: None if len(r["feature_names"]) > 40 else f"only {len(r['feature_names'])} features")
check("polynomial skipped when too wide", "train", bc, C | {"spec": {"kind": "single", "model": "logreg"}, "preprocess": {"poly_degree": 3}},
      lambda r: None if r["preprocessing"]["notes"] else "no skip note")
check("binning 5", "train", bc, C | {"spec": {"kind": "single", "model": "logreg"}, "preprocess": {"bin_numeric": 5}}, lambda r: None if any("bins" in s["title"] for s in r["preprocessing"]["steps"]) else "no bin step")
for rs in ("oversample", "undersample", "smote"):
    check(f"resample {rs}", "train", bc, C | {"spec": {"kind": "single", "model": "logreg"}, "preprocess": {"resample": rs}}, lambda r: None if r["balance"] is False else "balance still on")
check("resample in race", "leaderboard", bc, C | {"models": ["logreg", "rf"], "preprocess": {"resample": "smote"}})
check("grid search", "tune", bc, C | {"model": "rf", "n_iter": 8, "search": "grid"}, lambda r: None if r["search"] == "grid" and r["grid"] else "not grid")
check("random search still default", "tune", bc, C | {"model": "logreg", "n_iter": 4}, lambda r: None if r["search"] == "random" else "not random")
check("validation curve tree depth", "validation", bc, C | {"model": "tree", "param": "max_depth"}, lambda r: None if len(r["points"]) >= 3 else "few points")
check("validation curve default param", "validation", bc, C | {"model": "knn"})
for m in ("kbest", "mutual_info", "rfe", "model"):
    check(f"select {m}", "select", bc, C | {"method": m, "k": 8}, lambda r: None if len(r["selected"]) == 8 else f"{len(r['selected'])} selected")
check("calibration binary rf", "calibration", bc, C | {"model": "rf"}, lambda r: None if set(r["scores"]) == {"as trained", "sigmoid (Platt)", "isotonic"} else "missing variant")
model_id = tr and tr.get("model_id")
if model_id:
    open(os.path.join(work, "model.joblib"), "wb").write(call("GET", f"/api/models/{model_id}/download", raw=True))
    check("explain lgbm + one row", "explain", bc, {"model_id": model_id, "row_index": 3},
          lambda r: None if r["partial_dependence"] and r["local"] and r["local"]["contributions"] else "missing parts")

# Multiclass and regression
csv_for(wine)
check("calibration multiclass", "calibration", wine, W | {"model": "nb"})
check("xgboost multiclass", "train", wine, W | {"spec": {"kind": "single", "model": "xgb"}})
check("smote multiclass", "train", wine, W | {"spec": {"kind": "single", "model": "logreg"}, "preprocess": {"resample": "smote"}})
csv_for(dia)
check("xgboost regression", "train", dia, R | {"spec": {"kind": "single", "model": "xgb"}})
lg = check("lightgbm regression", "train", dia, R | {"spec": {"kind": "single", "model": "lgbm"}})
check("validation curve ridge alpha", "validation", dia, R | {"model": "ridge", "param": "alpha"})
check("select regression rfe", "select", dia, R | {"method": "rfe", "k": 4})
check("grid search regression", "tune", dia, R | {"model": "lgbm", "n_iter": 8, "search": "grid"})
check("auto still works", "auto", dia, R)
if lg and lg.get("model_id"):
    open(os.path.join(work, "model.joblib"), "wb").write(call("GET", f"/api/models/{lg['model_id']}/download", raw=True))
    check("explain regression", "explain", dia, {"model_id": lg["model_id"], "features": ["bmi", "bp"]})

# Guard rails
for label, kind, did, params in [
    ("calibration rejects regression", "calibration", dia, R | {"model": "ridge"}),
    ("explain rejects other dataset", "explain", dia, {"model_id": model_id or "x"}),
]:
    try:
        j, _ = run(kind, did, params)
        ok = j["status"] == "error"
    except SystemExit as e:
        ok = "400" in str(e) or "404" in str(e)
    print(f"{'ok  ' if ok else 'FAIL'} {label}")
    if not ok:
        fails.append(label)

print("\nALL PASSED" if not fails else f"\nFAILED: {fails}")
sys.exit(1 if fails else 0)
