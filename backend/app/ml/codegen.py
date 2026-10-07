"""Turn a finished job into one complete, runnable Python script.
The goal is understanding: the script mirrors exactly what the server did, with comments."""
from __future__ import annotations

import json

from . import catalog

SCALER = {"standard": "StandardScaler()", "minmax": "MinMaxScaler()", "robust": "RobustScaler()", "none": None}


def _prep_block(prep_info: dict) -> str:
    o = prep_info.get("options", {})
    cols = prep_info.get("columns", {})
    scaler = SCALER.get(o.get("scaler", "standard"))
    enc = "OneHotEncoder(handle_unknown='ignore', sparse_output=False)" if o.get("encoder", "onehot") == "onehot" else "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"
    num = f"[('impute', SimpleImputer(strategy='{o.get('num_impute', 'median')}'))" + (f", ('scale', {scaler})" if scaler else "") + "]"
    lines = [
        "# --- Preprocessing -----------------------------------------------------",
        "# Numeric columns: fill blanks, then scale. Text columns: fill blanks, then encode.",
        "# Wrapped in a ColumnTransformer so it is fitted on training rows only (no leakage).",
        f"numeric = {cols.get('numeric', [])!r}",
        f"categorical = {cols.get('categorical', [])!r}",
    ]
    dropped = [d["column"] for d in cols.get("dropped", [])]
    if dropped:
        lines.append(f"# Dropped automatically: {dropped!r}")
    lines += [
        "preprocessor = ColumnTransformer([",
        f"    ('num', Pipeline({num}), numeric),",
        f"    ('cat', Pipeline([('impute', SimpleImputer(strategy='{o.get('cat_impute', 'most_frequent')}', fill_value='missing')), ('encode', {enc})]), categorical),",
        "], remainder='drop')",
    ]
    return "\n".join(lines)


IMPORTS = """import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
"""


def estimator_code(task: str, spec: dict) -> tuple[str, str]:
    """Return (constructor expression, import lines) for a single model or an ensemble spec."""
    reg = catalog.registry(task)

    def one(key: str, params: dict | None = None) -> tuple[str, str]:
        meta = reg.get(key) or reg["rf"]
        est = meta["make"]()
        cls = est.__class__
        defaults = cls().get_params()
        shown = {k: v for k, v in est.get_params().items() if k in defaults and defaults[k] != v and k != "n_jobs"}
        for k, v in (params or {}).items():
            if k in defaults and v not in (None, ""):
                shown[k] = v
        args = ", ".join(f"{k}={v!r}" for k, v in shown.items())
        return f"{cls.__name__}({args})", f"from {cls.__module__.split('._')[0]} import {cls.__name__}\n"

    kind = spec.get("kind", "single")
    if kind == "single":
        return one(spec.get("model") or "rf", spec.get("params"))
    suffix = "Classifier" if task == "classification" else "Regressor"
    members = [k for k in spec.get("members", []) if k in reg]
    soft_svm = kind == "voting" and task == "classification" and spec.get("voting", "soft") == "soft"
    parts = [one(k, {"probability": True} if (soft_svm and k == "svm") else None) for k in members]
    imports = "".join(dict.fromkeys(p[1] for p in parts))
    inner = ",\n    ".join(f"('{k}', {parts[i][0]})" for i, k in enumerate(members))
    if kind == "voting":
        v = f", voting='{spec.get('voting', 'soft')}'" if task == "classification" else ""
        return f"Voting{suffix}([\n    {inner}\n]{v})", imports + f"from sklearn.ensemble import Voting{suffix}\n"
    if kind == "stacking":
        final, fimp = one(spec.get("final") or ("logreg" if task == "classification" else "ridge"))
        return f"Stacking{suffix}([\n    {inner}\n], final_estimator={final}, cv=3)", imports + fimp + f"from sklearn.ensemble import Stacking{suffix}\n"
    if kind == "bagging":
        base, bimp = one(spec.get("model") or "tree")
        return f"Bagging{suffix}(estimator={base}, n_estimators={int(spec.get('n_estimators', 25))}, random_state=42)", bimp + f"from sklearn.ensemble import Bagging{suffix}\n"
    return one("rf")


def _head(target: str | None, drop: list[str]) -> str:
    s = "df = pd.read_csv('your_dataset.csv')\n"
    if target:
        s += f"df = df[df['{target}'].notna()]\n"
        s += f"X = df.drop(columns={[target] + drop!r})\ny = df['{target}']\n"
    else:
        s += f"X = df.drop(columns={drop!r}) if {drop!r} else df\n"
    return s


def script(kind: str, params: dict, result: dict) -> str:
    task = params.get("task")
    target = params.get("target")
    prep_info = result.get("preprocessing", {})
    drop = prep_info.get("options", {}).get("drop_columns", []) or []
    head = _head(target, drop)
    prep = _prep_block(prep_info)
    if kind == "auto":
        race = script("leaderboard", params, result)
        best = result.get("best", {})
        est, imports = estimator_code(task, best.get("spec") or {"kind": "single", "model": "rf"})
        strat = ", stratify=y" if task == "classification" else ""
        top = [r for r in result.get("leaderboard", []) if r.get("status") == "ok" and r.get("family") != "ensemble"][:3]
        vcode, vimp = estimator_code(task, {"kind": "voting", "members": [r["key"] for r in top], "voting": "soft"})
        scode, simp = estimator_code(task, {"kind": "stacking", "members": [r["key"] for r in top]})
        return race + f"""
# --- Ensembles of the top three models -------------------------------------
{"".join(dict.fromkeys((vimp + simp + imports).splitlines(True)))}
ensembles = {{
    'Voting of top 3': {vcode},
    'Stacking of top 3': {scode},
}}
for name, model in ensembles.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='{result.get('scoring')}')
    print(f'{{name:32s}} {{scores.mean():.4f}} +/- {{scores.std():.4f}}')

# --- Train the overall winner: {best.get('name')} ---------------------------
from sklearn.model_selection import train_test_split
model = {est}
pipe = Pipeline([('prep', preprocessor), ('model', model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42{strat})
pipe.fit(X_train, y_train)
print('score on unseen rows:', pipe.score(X_test, y_test))
import joblib
joblib.dump(pipe, 'best_model.joblib')
"""
    if kind == "leaderboard":
        reg = catalog.registry(task)
        keys = [r["key"] for r in result.get("leaderboard", []) if r.get("status") == "ok" and r["key"] in reg]
        imports = "".join(dict.fromkeys(estimator_code(task, {"kind": "single", "model": k})[1] for k in keys))
        shelf = ",\n".join(f"    '{reg[k]['name']}': {estimator_code(task, {'kind': 'single', 'model': k})[0]}" for k in keys)
        cv = "StratifiedKFold" if task == "classification" else "KFold"
        return f"""{IMPORTS}from sklearn.model_selection import cross_val_score, {cv}
{imports}
{head}
{prep}

# --- The model shelf -----------------------------------------------------
models = {{
{shelf}
}}

# --- Race: {result.get('folds', 3)}-fold cross-validation, same folds for every model ----
cv = {cv}(n_splits={result.get('folds', 3)}, shuffle=True, random_state=42)
results = {{}}
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='{result.get('scoring')}')
    results[name] = (scores.mean(), scores.std())
    print(f'{{name:32s}} {{scores.mean():.4f}} +/- {{scores.std():.4f}}')

best = max(results, key=lambda n: results[n][0])
print('winner:', best)
"""
    if kind == "train":
        spec = params.get("spec") or {"kind": "single", "model": params.get("model")}
        est, imports = estimator_code(task, spec)
        strat = ", stratify=y" if task == "classification" else ""
        if task == "classification":
            metrics = """from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
pred = pipe.predict(X_test)
print('train accuracy:', accuracy_score(y_train, pipe.predict(X_train)))
print('test accuracy :', accuracy_score(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))"""
        else:
            metrics = """from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
pred = pipe.predict(X_test)
print('train R2:', r2_score(y_train, pipe.predict(X_train)))
print('test R2 :', r2_score(y_test, pred))
print('MAE     :', mean_absolute_error(y_test, pred))
print('RMSE    :', root_mean_squared_error(y_test, pred))"""
        return f"""{IMPORTS}from sklearn.model_selection import train_test_split
{imports}
{head}
{prep}

# --- Model ----------------------------------------------------------------
model = {est}
pipe = Pipeline([('prep', preprocessor), ('model', model)])

# --- Hold-out split: the model never sees X_test while learning -----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={result.get('test_size', 0.2)}, random_state=42{strat})
pipe.fit(X_train, y_train)

# --- Evaluate -------------------------------------------------------------
{metrics}

# --- Keep it ---------------------------------------------------------------
import joblib
joblib.dump(pipe, 'model.joblib')
"""
    if kind == "tune":
        est, imports = estimator_code(task, {"kind": "single", "model": params.get("model")})
        space = _space_literal(task, params.get("model"))
        cv = "StratifiedKFold" if task == "classification" else "KFold"
        return f"""{IMPORTS}from sklearn.model_selection import RandomizedSearchCV, {cv}
from scipy.stats import loguniform, randint, uniform
{imports}
{head}
{prep}

pipe = Pipeline([('prep', preprocessor), ('model', {est})])

# --- Search space: 'model__' prefixes reach into the pipeline step ---------
space = {space}

# --- Random search: {len(result.get('trials', []))} random combinations, each cross-validated ----------
search = RandomizedSearchCV(pipe, space, n_iter={len(result.get('trials', []))}, cv={cv}(n_splits=3, shuffle=True, random_state=42),
                            scoring='{result.get('scoring')}', random_state=42, n_jobs=1)
search.fit(X, y)
print('best score :', search.best_score_)
print('best params:', search.best_params_)
# Server found: {json.dumps(result.get('best_params', {}))}
"""
    if kind == "curve":
        spec = params.get("spec") or {"kind": "single", "model": params.get("model")}
        est, imports = estimator_code(task, spec)
        return f"""{IMPORTS}from sklearn.model_selection import learning_curve
{imports}
{head}
{prep}

pipe = Pipeline([('prep', preprocessor), ('model', {est})])

# --- Learning curve: retrain on growing slices, watch train vs test score ---
sizes, train_scores, test_scores = learning_curve(
    pipe, X, y, cv=3, scoring='{result.get('scoring')}',
    train_sizes=np.linspace(0.15, 1.0, 5), shuffle=True, random_state=42)
for n, tr, te in zip(sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)):
    print(f'{{n:6d}} rows  train {{tr:.4f}}  test {{te:.4f}}')
# Wide gap that stays wide = variance (overfitting). Two low curves = bias (underfitting).
"""
    if kind == "cluster":
        algo = params.get("algorithm", "kmeans")
        k = int(params.get("k", 3))
        ctor = {
            "kmeans": f"KMeans(n_clusters={k}, n_init=10, random_state=42)",
            "dbscan": f"DBSCAN(eps={float(params.get('eps', 0.5))}, min_samples={int(params.get('min_samples', 5))})",
            "agglomerative": f"AgglomerativeClustering(n_clusters={k}, linkage='{params.get('linkage', 'ward')}')",
            "gmm": f"GaussianMixture(n_components={k}, random_state=42)",
        }[algo]
        imp = {"kmeans": "from sklearn.cluster import KMeans", "dbscan": "from sklearn.cluster import DBSCAN", "agglomerative": "from sklearn.cluster import AgglomerativeClustering", "gmm": "from sklearn.mixture import GaussianMixture"}[algo]
        sweep = ""
        if algo == "kmeans":
            sweep = """
# --- Choosing k: elbow (inertia) and silhouette for k = 2..10 ---------------
for kk in range(2, 11):
    km = KMeans(n_clusters=kk, n_init=4, random_state=42).fit(Z)
    print(kk, 'inertia', round(km.inertia_, 1), 'silhouette', round(silhouette_score(Z, km.labels_), 3))
"""
        fit = "labels = model.fit_predict(Z)" if algo != "gmm" else "labels = model.fit(Z).predict(Z)"
        return f"""{IMPORTS}from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
{imp}

{head}
{prep}
# Clustering has no y. Fit the preprocessing on everything; distances need scaled features.
Z = preprocessor.fit_transform(X)
{sweep}
# --- Cluster -----------------------------------------------------------------
model = {ctor}
{fit}
print('cluster sizes:', pd.Series(labels).value_counts().sort_index().to_dict())
mask = labels != -1  # DBSCAN marks noise as -1
if len(set(labels[mask])) > 1:
    print('silhouette     :', silhouette_score(Z[mask], labels[mask]))
    print('davies-bouldin :', davies_bouldin_score(Z[mask], labels[mask]))

# --- Look at it in 2D ---------------------------------------------------------
xy = PCA(n_components=2).fit_transform(Z)
# plt.scatter(xy[:, 0], xy[:, 1], c=labels)
"""
    if kind == "reduce":
        algo = params.get("algorithm", "pca")
        if algo == "pca":
            n = int(params.get("n_components", 2))
            return f"""{IMPORTS}from sklearn.decomposition import PCA

{head}
{prep}
Z = preprocessor.fit_transform(X)

# --- PCA: new axes ordered by explained variance -----------------------------
pca = PCA(n_components={n}, random_state=0)
T = pca.fit_transform(Z)
print('explained variance ratio:', pca.explained_variance_ratio_.round(3))
print('cumulative              :', np.cumsum(pca.explained_variance_ratio_).round(3))
# Loadings: which original features make up each component
names = preprocessor.get_feature_names_out()
for i, comp in enumerate(pca.components_[:2]):
    top = np.argsort(np.abs(comp))[::-1][:8]
    print(f'PC{{i+1}}:', [(names[j], round(comp[j], 3)) for j in top])
# plt.scatter(T[:, 0], T[:, 1]{", c=df['" + params['color_by'] + "'].astype('category').cat.codes" if params.get('color_by') else ''})
"""
        return f"""{IMPORTS}from sklearn.manifold import TSNE

{head}
{prep}
Z = preprocessor.fit_transform(X)
if len(Z) > 1500:  # t-SNE is O(n^2); keep it small
    idx = np.random.RandomState(0).choice(len(Z), 1500, replace=False)
    Z = Z[idx]

# --- t-SNE: a 2D map that keeps neighbours close -----------------------------
T = TSNE(n_components=2, perplexity={float(params.get('perplexity', 30))}, init='pca', random_state=0).fit_transform(Z)
# plt.scatter(T[:, 0], T[:, 1])
"""
    if kind == "anomaly":
        algo = params.get("algorithm", "isoforest")
        c = float(params.get("contamination", 0.05))
        if algo == "isoforest":
            body = f"""model = IsolationForest(contamination={c}, random_state=42)
flag = model.fit_predict(Z) == -1      # -1 means anomaly
score = -model.score_samples(Z)         # higher = more anomalous"""
            imp = "from sklearn.ensemble import IsolationForest"
        else:
            body = f"""model = LocalOutlierFactor(n_neighbors=20, contamination={c})
flag = model.fit_predict(Z) == -1
score = -model.negative_outlier_factor_"""
            imp = "from sklearn.neighbors import LocalOutlierFactor"
        return f"""{IMPORTS}{imp}

{head}
{prep}
Z = preprocessor.fit_transform(X)

# --- Anomaly detection: flag roughly {c*100:.1f}% of rows -----------------------------
{body}
print('flagged:', flag.sum())
print(X[flag].assign(score=score[flag]).sort_values('score', ascending=False).head(25))
"""
    return "# no code generator for this job kind"


def _space_literal(task: str, key: str) -> str:
    meta = catalog.registry(task).get(key)
    if not meta:
        return "{}"
    items = []
    for k, v in meta["space"].items():
        if isinstance(v, list):
            items.append(f"    '{k}': {v!r}")
        else:
            d = getattr(v, "dist", None)
            name = getattr(d, "name", "") if d is not None else ""
            a = v.args if hasattr(v, "args") else ()
            if name == "loguniform" or "loguniform" in str(type(d)).lower():
                items.append(f"    '{k}': loguniform({a[0]!r}, {a[1]!r})")
            elif name == "randint":
                items.append(f"    '{k}': randint({a[0]!r}, {a[1]!r})")
            elif name == "uniform":
                items.append(f"    '{k}': uniform({a[0]!r}, {a[1]!r})")
            else:
                items.append(f"    '{k}': <distribution>")
    return "{\n" + ",\n".join(items) + "\n}"
