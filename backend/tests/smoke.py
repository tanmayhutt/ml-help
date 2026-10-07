"""End-to-end smoke test against a running server: python tests/smoke.py http://127.0.0.1:8000"""
import json, sys, time, urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def wait(job_id, timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = call("GET", f"/api/jobs/{job_id}")
        if j["status"] in ("done", "error"):
            return j
        time.sleep(1)
    raise SystemExit("timeout")


print(call("GET", "/api/health"))
ds = call("POST", "/api/datasets/sample/iris"); did = ds["id"]
print("dataset", did, call("POST", f"/api/datasets/{did}/task", {"target": "species"}))
jobs = [
    ("leaderboard", {"task": "classification", "target": "species", "include_slow": True}),
    ("train", {"task": "classification", "target": "species", "spec": {"kind": "stacking", "members": ["logreg", "knn", "rf"], "final": "logreg"}}),
    ("train", {"task": "classification", "target": "species", "spec": {"kind": "voting", "members": ["logreg", "svm", "rf"], "voting": "soft"}}),
    ("tune", {"task": "classification", "target": "species", "model": "rf", "n_iter": 4}),
    ("curve", {"task": "classification", "target": "species", "model": "logreg"}),
    ("cluster", {"algorithm": "kmeans", "k": 3, "preprocess": {"drop_columns": ["species"]}}),
    ("cluster", {"algorithm": "dbscan", "eps": 0.6, "preprocess": {"drop_columns": ["species"]}}),
    ("reduce", {"algorithm": "pca", "color_by": "species", "preprocess": {"drop_columns": ["species"]}}),
    ("reduce", {"algorithm": "tsne", "color_by": "species", "preprocess": {"drop_columns": ["species"]}}),
    ("anomaly", {"algorithm": "isoforest", "preprocess": {"drop_columns": ["species"]}}),
]
model_id = None
for kind, params in jobs:
    t0 = time.time()
    j = wait(call("POST", "/api/jobs", {"kind": kind, "dataset_id": did, "params": params})["id"])
    print(f"{kind:12s} {j['status']:6s} {time.time()-t0:5.1f}s", j.get("error") or (j["result"].get("summary") or j["result"].get("model_name") or j["result"].get("algorithm")))
    if j["status"] == "done" and j["result"].get("model_id"):
        model_id = j["result"]["model_id"]
dd = call("POST", "/api/datasets/sample/diabetes")["id"]
j = wait(call("POST", "/api/jobs", {"kind": "train", "dataset_id": dd, "params": {"task": "regression", "target": "progression", "model": "gb"}})["id"])
print("regression", j["status"], j.get("error") or j["result"]["evaluation"]["metrics"])
if model_id:
    print("predict", call("POST", f"/api/models/{model_id}/predict", {"rows": [{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}]}))
    with urllib.request.urlopen(BASE + f"/api/models/{model_id}/notebook") as r:
        print("notebook bytes", len(r.read()))
