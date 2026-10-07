# ML Help

Upload a dataset, pick a column, and find the best scikit-learn model in numbered steps. Combine models, tune, cluster, draw a 2D map, flag unusual rows. Every result comes with a plain-language explanation and the full Python code.

## Run locally

```bash
cd backend
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000. The frontend is plain HTML, CSS, and ES modules in `frontend/`, served by FastAPI. No build step.

End-to-end check against a running server:

```bash
cd backend && .venv/bin/python tests/smoke.py http://127.0.0.1:8000
```

## Layout

- `backend/app/config.py`: every resource cap, all overridable by `ML_*` environment variables.
- `backend/app/runner.py`: one job at a time, each in a spawned process with a hard timeout and memory limit.
- `backend/app/ml/`: catalog (models with explanations), preprocess, train (race, train, ensembles, tune, curve), unsupervised, evaluate, codegen, export.
- `frontend/static/`: `ui.js` DOM helpers, `charts.js` SVG charts, `views/` one module per screen.
- `deploy/`: Caddy snippet and deploy script for the shared server.

## Docker

```bash
docker compose up --build
```

Serves on 127.0.0.1:5080 with a 2 CPU, 3 GB ceiling.
