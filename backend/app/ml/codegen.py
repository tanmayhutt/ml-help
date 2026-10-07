"""Turn a finished job into one complete, runnable Python script.
The goal is understanding: the script mirrors exactly what the server did, with comments."""
from __future__ import annotations

import json

from . import catalog

SCALER = {"standard": "StandardScaler()", "minmax": "MinMaxScaler()", "robust": "RobustScaler()", "none": None}


DATE_EXPANDER = """
class DateExpander(BaseEstimator, TransformerMixin):
    # Date text -> year, month, weekday, day of year, days since the earliest training date.
    def __init__(self, columns=()):
        self.columns = columns
    def fit(self, X, y=None):
        self.origin_ = {c: pd.to_datetime(X[c], errors='coerce', format='mixed').min() for c in self.columns}
        return self
    def transform(self, X):
        X = X.copy()
        for c in self.columns:
            d = pd.to_datetime(X[c], errors='coerce', format='mixed')
            X[c + '_year'], X[c + '_month'], X[c + '_weekday'], X[c + '_dayofyear'] = d.dt.year, d.dt.month, d.dt.weekday, d.dt.dayofyear
            X[c + '_days'] = (d - self.origin_[c]).dt.days
            X = X.drop(columns=[c])
        return X
"""

WINSORIZER = """
class Winsorizer(BaseEstimator, TransformerMixin):
    # Cap values outside [Q1 - k*IQR, Q3 + k*IQR]. Limits are learned on training rows only.
    def __init__(self, k=1.5):
        self.k = k
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        self.lower_, self.upper_ = q1 - self.k * (q3 - q1), q3 + self.k * (q3 - q1)
        return self
    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)
"""


def _prep_block(prep_info: dict) -> str:
    o = prep_info.get("options", {})
    cols = prep_info.get("columns", {})
    scaler = SCALER.get(o.get("scaler", "standard"))
    rare = int(o.get("rare_min") or 0)
    enc = (f"OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency={rare}, sparse_output=False)" if rare > 1 else "OneHotEncoder(handle_unknown='ignore', sparse_output=False)") if o.get("encoder", "onehot") == "onehot" else "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)"
    cap = o.get("outliers") == "cap"
    base = f"('impute', SimpleImputer(strategy='{o.get('num_impute', 'median')}'))" + (", ('cap', Winsorizer())" if cap else "")
    num = "[" + base + (f", ('scale', {scaler})" if scaler else "") + "]"
    num_log = "[" + base + ", ('log', FunctionTransformer(np.log1p))" + (f", ('scale', {scaler})" if scaler else "") + "]"
    lines = [
        "# --- Preprocessing -----------------------------------------------------",
        "# Numeric columns: fill blanks" + (", cap outliers" if cap else "") + ", then scale. Text columns: fill blanks, then encode.",
        "# Wrapped in a ColumnTransformer so it is fitted on training rows only (no leakage).",
    ]
    if o.get("drop_duplicates"):
        lines.append("df = df.drop_duplicates()")
    if cap:
        lines.append(WINSORIZER)
    if cols.get("dates"):
        lines.append(DATE_EXPANDER)
    dropped_corr = [d["column"] for d in cols.get("dropped", []) if "nearly identical" in d.get("reason", "")]
    if dropped_corr:
        lines.append(f"# Dropped as near-duplicates of other columns (correlation above {o.get('drop_correlated', 0.95)}): {dropped_corr!r}")
    lines += [
        f"numeric = {cols.get('numeric', [])!r}",
        f"categorical = {cols.get('categorical', [])!r}",
    ]
    if cols.get("numeric_log"):
        lines.append(f"numeric_log = {cols.get('numeric_log')!r}  # skewed columns get log(1 + x)")
    if cols.get("ordered"):
        lines.append(f"ordered = {cols.get('ordered')!r}  # categories with a natural order")
    if cols.get("dates"):
        lines.append(f"dates = {cols.get('dates')!r}")
    dropped = [d["column"] for d in cols.get("dropped", [])]
    if dropped:
        lines.append(f"# Dropped automatically: {dropped!r}")
    lines += [
        "preprocessor = ColumnTransformer([",
        f"    ('num', Pipeline({num}), numeric),",
    ]
    if cols.get("numeric_log"):
        lines.append(f"    ('num_log', Pipeline({num_log}), numeric_log),")
    lines.append(f"    ('cat', Pipeline([('impute', SimpleImputer(strategy='{o.get('cat_impute', 'most_frequent')}', fill_value='missing')), ('encode', {enc})]), categorical),")
    if cols.get("ordered"):
        lines.append("    ('ord', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OrdinalEncoder(categories=[ordered[c] for c in ordered])), ('scale', StandardScaler())]), list(ordered)),")
    for c in cols.get("dates", []):
        lines.append(f"    ('date_{c}', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), [f'{c}_{{p}}' for p in ('year', 'month', 'weekday', 'dayofyear', 'days')]),")
    lines.append("], remainder='drop')")
    if cols.get("dates"):
        lines.append("preprocessor = Pipeline([('dates', DateExpander(dates)), ('columns', preprocessor)])")
    if cols.get("ordered"):
        lines.append("for c in ordered:\n    X[c] = X[c].astype(str).str.strip().str.lower()")
    return "\n".join(lines)


IMPORTS = """import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
"""


def estimator_code(task: str, spec: dict, balance: bool = False, log_target: bool = False) -> tuple[str, str]:
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
        if balance and key in catalog.BALANCEABLE and "class_weight" in defaults:
            shown["class_weight"] = "balanced"
        args = ", ".join(f"{k}={v!r}" for k, v in shown.items())
        return f"{cls.__name__}({args})", f"from {cls.__module__.split('._')[0]} import {cls.__name__}\n"

    code, imports = _estimator_inner(task, spec, reg, one)
    if log_target and task == "regression":
        code = f"TransformedTargetRegressor(regressor={code}, func=np.log1p, inverse_func=np.expm1)"
        imports += "from sklearn.compose import TransformedTargetRegressor\n"
    return code, imports


def _estimator_inner(task: str, spec: dict, reg: dict, one) -> tuple[str, str]:

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
    balance = bool(result.get("balance"))
    log_target = bool(result.get("log_target"))
    head = _head(target, drop)
    prep = _prep_block(prep_info)
    if kind == "auto":
        race = script("leaderboard", params, result)
        best = result.get("best", {})
        est, imports = estimator_code(task, best.get("spec") or {"kind": "single", "model": "rf"}, balance, log_target)
        strat = ", stratify=y" if task == "classification" else ""
        top = [r for r in result.get("leaderboard", []) if r.get("status") == "ok" and r.get("family") != "ensemble"][:3]
        vcode, vimp = estimator_code(task, {"kind": "voting", "members": [r["key"] for r in top], "voting": "soft"}, balance, log_target)
        scode, simp = estimator_code(task, {"kind": "stacking", "members": [r["key"] for r in top]}, balance, log_target)
        refine = "".join(f"# - {r['step']}: {'applied' if r.get('applied') else 'not applied'}" + (f" ({r['before']} -> {r['after']})" if r.get('before') is not None and r.get('after') is not None else "") + "\n" for r in result.get("refinements", []))
        thr = result.get("threshold")
        thr_code = f"\n# Decision threshold chosen for the rare class:\nproba = pipe.predict_proba(X_test)[:, 1]\npred = (proba >= {thr}).astype(int)\n" if thr else ""
        return race + f"""
# --- Refinements that were tested on the winner ------------------------------
{refine}
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
print('score on unseen rows:', pipe.score(X_test, y_test)){thr_code}
import joblib
joblib.dump(pipe, 'best_model.joblib')
"""
    if kind == "leaderboard":
        reg = catalog.registry(task)
        keys = [r["key"] for r in result.get("leaderboard", []) if r.get("status") == "ok" and r["key"] in reg]
        imports = "".join(dict.fromkeys(estimator_code(task, {"kind": "single", "model": k}, balance, log_target)[1] for k in keys))
        shelf = ",\n".join(f"    '{reg[k]['name']}': {estimator_code(task, {'kind': 'single', 'model': k}, balance, log_target)[0]}" for k in keys)
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
        est, imports = estimator_code(task, spec, balance, log_target)
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
        est, imports = estimator_code(task, {"kind": "single", "model": params.get("model")}, balance, log_target)
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
        est, imports = estimator_code(task, spec, balance, log_target)
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
