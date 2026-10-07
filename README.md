# ML Help

Upload a table, pick a column, and find the best model for it. No machine learning background needed.

Live: https://ml.tanmaytiwari.me

ML Help walks you through machine learning in numbered steps. Every result comes with a one-line answer in plain words, a step-by-step explanation of what was done, and the complete Python code to reproduce it with scikit-learn.

## What it does

**Understand and clean your data**

- Overview, every column with histogram and box plot, category bars, missing-values chart, correlation heatmap, the most related pairs, duplicate count.
- A generated cleaning plan: which columns to drop and why, duplicates, how to fill blanks, outlier capping, log transform for long tails, encoding per column, scaling. Each step has the reason and pandas code, and one button applies the plan to the prediction flow.
- The seaborn and matplotlib code for every chart.

**Predict a column**

- One button finds the best model: 13 classifiers or 13 regressors are cross-validated on the same folds, the top three are combined by voting and stacking, the overall winner is trained on a hold-out split, and a plain-language verdict explains the choice.
- Trains one model with full metrics and charts: confusion matrix, ROC curve, predicted vs actual, error histogram, feature importance.
- Combines models: voting, stacking, and bagging ensembles built from any members you pick.
- Fine-tunes a model with random search over its hyperparameters.
- Checks whether more data would help with a learning curve.
- Saves every trained model so you can predict new rows, download the model, or download a notebook.

**Find groups or odd rows** (no target column)

- Clustering: K-Means with an automatic sweep over the number of groups, DBSCAN, agglomerative, Gaussian mixture.
- 2D pictures: PCA with loadings, t-SNE.
- Unusual rows: Isolation Forest, Local Outlier Factor.

**Every run shows its work**

- Data preparation is explained step by step: how blanks were filled, how numbers were scaled, how text was encoded, which columns were dropped and why.
- Each metric and chart has a short note on how to read it.
- A full, runnable Python script is attached to every result.

## Run it locally

Requires Python 3.12. The frontend is plain HTML, CSS and JavaScript served by the backend, so there is no build step.

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000.

Or with Docker:

```bash
docker compose up --build
```

Then open http://127.0.0.1:5080.

End-to-end check against a running server:

```bash
cd backend && .venv/bin/python tests/smoke.py http://127.0.0.1:8000
```

## How it stays cheap to host

The app is designed to run on a small shared server without surprising bills.

- One job runs at a time. Each job runs in its own throwaway process with a hard time limit and a memory cap, so a runaway job is killed and memory returns to baseline.
- Large files are subsampled for the model race, tuning, and t-SNE. Slow models are skipped on big files unless you ask for them.
- Uploads are capped in size, rows, and columns, and rate limited per client.
- Uploaded files and models are deleted automatically after a retention window.
- No Redis, no Postgres, no background workers. SQLite plus files on disk. Idle cost is one sleeping process.

Every limit is an environment variable. See `backend/app/config.py`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ML_JOB_TIMEOUT_SEC` | 180 | Hard kill for any job |
| `ML_JOB_MEMORY_MB` | 2048 | Memory cap per job process |
| `ML_N_JOBS` | 2 | CPU threads scikit-learn may use |
| `ML_QUEUE_MAX` | 6 | Jobs allowed to wait |
| `ML_MAX_UPLOAD_MB` | 20 | Upload size limit |
| `ML_MAX_ROWS` | 200000 | Rows per file |
| `ML_LEADERBOARD_ROWS` | 5000 | Rows used in the model race |
| `ML_TUNE_MAX_ITER` | 20 | Random-search trials |
| `ML_RETENTION_DAYS` | 7 | Days before files and models are deleted |
| `ML_ACCESS_TOKEN` | unset | If set, every request must send this token |

## Project layout

```
backend/app/
  main.py        FastAPI app, serves the API and the static frontend
  api.py         All HTTP routes
  runner.py      One-at-a-time job runner, each job in a killable subprocess
  config.py      Every limit, read from environment variables
  profile.py     Dataset profiling with plain-language notes
  ml/catalog.py  The model shelf: constructors, search spaces, explanations
  ml/preprocess.py, train.py, unsupervised.py, evaluate.py
  ml/codegen.py  Turns any finished job into a full Python script
frontend/
  index.html, static/app.css
  static/app.js        Hash router
  static/ui.js         DOM helpers and shared components
  static/charts.js     Hand-drawn SVG and HTML charts
  static/views/        One module per screen
deploy/
  Dockerfile, compose.yaml, Caddy snippet, deploy scripts
.github/workflows/deploy.yml   Deploys on push to main
```

## Stack

FastAPI, pandas, scikit-learn, SQLite, vanilla JavaScript. Docker behind Caddy in production, deployed by GitHub Actions over a restricted SSH key.

## License

MIT
