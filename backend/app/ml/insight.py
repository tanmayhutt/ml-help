"""Jobs that look inside a model instead of only scoring it:
validation curve (one setting from too simple to too complex), explain (partial dependence
and what-if contributions for one row), feature selection, and probability calibration.
Same contract as train.py: (df, params, progress, deadline) -> dict with narration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_predict, cross_val_score, validation_curve

from .. import config
from . import catalog, preprocess
from .train import _cv, _r, _scoring, build_estimator, make_model, make_pipe, prepare, values_from_dist

TEXT = {
    "validation": "A validation curve trains the same model again and again while one setting moves from simple to complex. On the simple side both scores are low (underfitting, high bias). On the complex side the training score keeps rising while the score on new rows falls (overfitting, high variance). The best setting sits where the score on new rows peaks.",
    "explain": "Partial dependence: set one column to the same value in every row, ask the model, and average its answers. Repeat across the column's range to see how the prediction moves with that column alone. A what-if contribution for one row resets one column to a typical value and measures how much the prediction changes.",
    "select": "Feature selection keeps only the columns that carry signal. Fewer columns can mean a simpler, faster model that overfits less. It is done inside the pipeline so the test rows never influence which columns are picked.",
    "calibration": "A model is calibrated when its confidence matches reality: of all rows where it says 80%, about 80% really are positive. Trees and boosting are often overconfident or underconfident. Calibration learns a correction on top of the model.",
    "grid_vs_random": "Grid search tries every combination of a few fixed values. Random search samples combinations from ranges. With the same budget, random search usually finds good settings faster because it explores more distinct values of the settings that matter most.",
}


def _prefix(P: dict) -> str:
    return "model__regressor__" if P.get("log_target") else "model__"


# ---- validation curve ---------------------------------------------------------

def validation(df, params: dict, progress, deadline) -> dict:
    task, target, key = params["task"], params["target"], params.get("model")
    reg = catalog.registry(task)
    if key not in reg:
        raise ValueError(f"Unknown model '{key}'.")
    m = reg[key]
    tunable = {k.replace("model__", ""): v for k, v in m["space"].items()}
    name = params.get("param") or next((p for p, v in tunable.items() if not (isinstance(v, list) and v and isinstance(v[0], tuple))), None)
    if not name:
        raise ValueError(f"{m['name']} has no setting that a validation curve can move.")
    if name not in tunable and not params.get("values"):
        raise ValueError(f"'{name}' is not a setting of {m['name']}. Try one of: {', '.join(tunable)}.")
    values = params.get("values") or values_from_dist(tunable[name], 7)
    values = [v for v in values if not isinstance(v, (list, tuple))][:10]
    if len(values) < 2:
        raise ValueError("A validation curve needs at least two values.")
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, sub = preprocess.subsample(P["X"], P["y"], config.TUNE_ROWS, task == "classification")
    scoring = _scoring(task, params.get("metric"))
    folds = min(config.CV_FOLDS, int(params.get("folds") or config.CV_FOLDS))
    pipe = make_pipe(P["prep"], make_model(task, key, P), P)
    progress(f"Training {m['name']} at {len(values)} values of {name}", 0.1)
    tr, te = validation_curve(pipe, X, y, param_name=_prefix(P) + name, param_range=values, cv=_cv(task, folds), scoring=scoring, n_jobs=1, error_score=np.nan)
    points = [{"value": v if not isinstance(v, (np.generic,)) else v.item(), "train": _r(a.mean()), "test": _r(b.mean()), "test_std": _r(b.std())} for v, a, b in zip(values, tr, te)]
    ok = [p for p in points if p["test"] is not None]
    if not ok:
        raise ValueError("Every value failed to train.")
    best = max(ok, key=lambda p: p["test"])
    i = points.index(best)
    gap_best = (best["train"] or 0) - best["test"]
    verdict = [f"The best value of {name} is {best['value']}: {best['test']:.3f} on new rows."]
    if i > 0 and points[0]["test"] is not None and best["test"] - points[0]["test"] > 0.01:
        verdict.append(f"At {points[0]['value']} the model is too simple: it scores {points[0]['test']:.3f}. That side is underfitting.")
    last = points[-1]
    if i < len(points) - 1 and last["test"] is not None and best["test"] - last["test"] > 0.01:
        verdict.append(f"At {last['value']} the training score is {last['train']:.3f} but new rows get only {last['test']:.3f}. That side is overfitting.")
    if len(verdict) == 1:
        verdict.append("The score barely changes across this range, so this setting does not matter much for this data.")
    if gap_best > 0.1:
        verdict.append(f"Even at the best value the training score is {gap_best:.3f} higher than on new rows, so some overfitting remains.")
    return {
        "task": task, "target": target, "model": key, "model_name": m["name"], "param": name, "param_help": m["params"].get(name, ""),
        "scoring": scoring, "rows_used": len(X), "subsampled": sub, "points": points, "best_value": best["value"], "verdict": verdict,
        "preprocessing": P["prep_info"], "balance": P.get("balance"), "log_target": P.get("log_target"),
        "narration": [
            {"step": "Validation curve", "text": TEXT["validation"], "code": f"validation_curve(pipe, X, y, param_name='model__{name}', param_range={values!r}, cv={folds})"},
            {"step": "Reading it", "text": " ".join(verdict), "code": ""},
        ],
    }


# ---- explain a trained model ---------------------------------------------------

def _model_view(pipe, X: pd.DataFrame, task: str, classes: list | None, threshold=None):
    """Return a function rows -> number per row that the explanation tracks."""
    if task == "regression":
        return lambda Z: np.asarray(pipe.predict(Z), dtype=float), "predicted value"
    if classes and len(classes) == 2 and hasattr(pipe, "predict_proba"):
        return lambda Z: pipe.predict_proba(Z)[:, 1], f"probability of '{classes[1]}'"
    if hasattr(pipe, "predict_proba"):
        return lambda Z: pipe.predict_proba(Z).max(axis=1), "confidence in the predicted class"
    if classes and len(classes) == 2 and hasattr(pipe, "decision_function"):
        return lambda Z: np.asarray(pipe.decision_function(Z), dtype=float), f"decision score (above 0 means '{classes[1]}')"
    return lambda Z: np.asarray(pipe.predict(Z), dtype=float), "predicted class number"


def _typical(s: pd.Series):
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9:
        return float(num.median())
    mode = s.mode(dropna=True)
    return mode.iloc[0] if len(mode) else None


def _grid(s: pd.Series) -> tuple[list, bool]:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.9 and num.nunique() > 8:
        qs = np.unique(np.quantile(num.dropna(), np.linspace(0.05, 0.95, 12)))
        return [float(f"{q:.6g}") for q in qs], True
    return [v for v in s.astype(str).value_counts().index[:8]], False


def explain(df, params: dict, progress, deadline) -> dict:
    import joblib

    from .. import storage
    meta = params["_model"]  # filled in by the API from the saved model record
    task, target, classes = meta["task"], meta["target"], meta.get("classes")
    pipe = joblib.load(storage.model_path(params["model_id"]))
    feats = [f for f in meta["features"] if f in df.columns]
    data = df[df[target].notna()] if target in df.columns else df
    X = data[feats]
    Xs = X.sample(min(len(X), 400), random_state=0) if len(X) > 400 else X
    f, what = _model_view(pipe, Xs, task, classes)
    base_pred = f(Xs)
    progress("Measuring how much the model relies on each column", 0.1)
    reliance = []
    rng = np.random.RandomState(0)
    for c in feats:
        Z = Xs.copy()
        Z[c] = rng.permutation(Z[c].values)
        reliance.append((c, float(np.mean(np.abs(f(Z) - base_pred)))))
    reliance.sort(key=lambda t: -t[1])
    chosen = [c for c in (params.get("features") or []) if c in feats][:6] or [c for c, _ in reliance[:4]]
    pd_out = []
    for n, c in enumerate(chosen):
        progress(f"Partial dependence for {c}", 0.2 + 0.6 * n / max(len(chosen), 1))
        grid, numeric = _grid(X[c])
        avg, ice = [], []
        ice_rows = Xs.iloc[:8]
        lines = [[] for _ in range(len(ice_rows))]
        for v in grid:
            Z = Xs.copy()
            Z[c] = v
            avg.append(_r(f(Z).mean()))
            if numeric:
                Zi = ice_rows.copy()
                Zi[c] = v
                for j, p in enumerate(f(Zi)):
                    lines[j].append(_r(p))
        lo, hi = min(a for a in avg if a is not None), max(a for a in avg if a is not None)
        trend = ""
        if numeric and len(avg) > 2:
            d = np.diff([a for a in avg if a is not None])
            trend = "rises" if (d >= -1e-9).all() and hi - lo > 1e-6 else "falls" if (d <= 1e-9).all() and hi - lo > 1e-6 else "goes up and down"
        pd_out.append({"feature": c, "numeric": numeric, "grid": grid, "average": avg, "ice": lines if numeric else [],
                       "range": _r(hi - lo), "text": (f"As {c} increases, the {what} {trend}; it moves by {hi - lo:.3g} across the range." if numeric else
                                                      f"Across the values of {c}, the {what} moves by {hi - lo:.3g}.")})
    local = None
    row = params.get("row")
    if row is None and params.get("row_index") is not None:
        idx = int(params["row_index"])
        if 0 <= idx < len(X):
            row = X.iloc[idx].to_dict()
    if row is not None:
        progress("Explaining one row", 0.9)
        r = pd.DataFrame([{c: row.get(c) for c in feats}])
        for c in feats:
            if pd.api.types.is_numeric_dtype(X[c]):
                r[c] = pd.to_numeric(r[c], errors="coerce")
        p0 = float(f(r)[0])
        contrib = []
        for c in feats:
            t = _typical(X[c])
            if t is None:
                continue
            Z = r.copy()
            Z[c] = t
            contrib.append({"feature": c, "value": None if pd.isna(r[c].iloc[0]) else r[c].iloc[0], "typical": t, "effect": _r(p0 - float(f(Z)[0]))})
        contrib.sort(key=lambda d: -abs(d["effect"] or 0))
        pred = pipe.predict(r)[0]
        label = classes[int(pred)] if classes and str(pred).lstrip("-").isdigit() else (float(pred) if task == "regression" else str(pred))
        local = {"prediction": label, "tracked": _r(p0), "contributions": contrib[:12],
                 "text": f"The model gives {what} {p0:.3g} for this row. Each effect is how much that drops or rises when only that column is reset to a typical value. Positive means the column's actual value pushed the {what} up."}
    return {
        "task": task, "target": target, "model_name": meta.get("name"), "tracked": what, "rows_used": len(Xs),
        "reliance": [{"feature": c, "value": _r(v)} for c, v in reliance[:15]], "partial_dependence": pd_out, "local": local,
        "narration": [
            {"step": "Which columns matter", "text": f"Each column was shuffled on its own and the change in the {what} was measured. Big change = the model relies on that column. This needs no answer column, so it shows what the model uses, not whether it is right.", "code": "Z[c] = np.random.permutation(Z[c]); np.mean(abs(f(Z) - f(X)))"},
            {"step": "Partial dependence", "text": TEXT["explain"] + " The thin lines (ICE) are single rows: if they cross, the column's effect depends on other columns.", "code": "for v in grid: Z = X.copy(); Z[c] = v; avg.append(model.predict_proba(Z)[:, 1].mean())"},
        ] + ([{"step": "One row", "text": local["text"], "code": "Z = row.copy(); Z[c] = typical[c]; effect = f(row) - f(Z)"}] if local else []),
    }


# ---- feature selection ----------------------------------------------------------

def select(df, params: dict, progress, deadline) -> dict:
    task, target = params["task"], params["target"]
    method = params.get("method") or "kbest"
    reg = catalog.registry(task)
    key = params.get("model") or ("logreg" if task == "classification" else "ridge")
    if key not in reg:
        raise ValueError(f"Unknown model '{key}'.")
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, sub = preprocess.subsample(P["X"], P["y"], config.TUNE_ROWS, task == "classification")
    n_feat = clone(P["prep"]).fit(X, y).transform(X.iloc[:5]).shape[1]
    if n_feat < 3:
        raise ValueError("Feature selection needs at least 3 prepared columns.")
    k = int(params.get("k") or max(1, n_feat // 2))
    k = min(max(k, 1), n_feat - 1)
    scoring = _scoring(task, params.get("metric"))
    folds = min(config.CV_FOLDS, int(params.get("folds") or config.CV_FOLDS))
    clf = task == "classification"
    if method == "kbest":
        sel, how = SelectKBest(f_classif if clf else f_regression, k=k), f"SelectKBest({'f_classif' if clf else 'f_regression'}, k={k})"
        why = "Each column is scored on its own with an F-test: how strongly it differs between classes (or lines up with the answer). The top k are kept. Fast, but blind to columns that only help together."
    elif method == "mutual_info":
        fn = mutual_info_classif if clf else mutual_info_regression
        sel, how = SelectKBest(lambda A, b: fn(A, b, random_state=0), k=k), f"SelectKBest({fn.__name__}, k={k})"
        why = "Each column is scored by mutual information: how much knowing it reduces uncertainty about the answer. Unlike the F-test it also catches curved relationships."
    elif method == "rfe":
        base = make_model(task, "logreg" if clf else "ridge", {})
        sel, how = RFE(base, n_features_to_select=k, step=max(1, n_feat // 10)), f"RFE({base.__class__.__name__}(), n_features_to_select={k})"
        why = "Recursive feature elimination fits a model, drops the weakest columns, and repeats until k remain. It judges columns together, not one by one."
    elif method == "model":
        base = make_model(task, "rf", {})
        sel, how = SelectFromModel(base, max_features=k, threshold=-np.inf), f"SelectFromModel(RandomForest(), max_features={k})"
        why = "A random forest is trained once and its importance scores pick the top k columns."
    else:
        raise ValueError(f"Unknown method '{method}'.")
    est = make_model(task, key, P)
    progress("Scoring with every column", 0.15)
    cv = _cv(task, folds)
    all_scores = cross_val_score(make_pipe(clone(P["prep"]), clone(est), P), X, y, cv=cv, scoring=scoring, n_jobs=1)
    progress(f"Scoring with the best {k} columns", 0.45)
    sel_scores = cross_val_score(make_pipe(clone(P["prep"]), clone(est), P, middle=[("select", clone(sel))]), X, y, cv=cv, scoring=scoring, n_jobs=1)
    progress("Picking the columns on all rows", 0.8)
    full = make_pipe(clone(P["prep"]), clone(est), P, middle=[("select", clone(sel))]).fit(X, y)
    names = preprocess.feature_names(full.named_steps["prep"]) or [f"f{i}" for i in range(n_feat)]
    s = full.named_steps["select"]
    mask = s.get_support()
    ranking = []
    if hasattr(s, "scores_") and s.scores_ is not None:
        ranking = sorted(({"feature": names[i], "score": _r(v)} for i, v in enumerate(s.scores_)), key=lambda d: -(d["score"] or 0))[:25]
    elif hasattr(s, "ranking_"):
        ranking = sorted(({"feature": names[i], "rank": int(v)} for i, v in enumerate(s.ranking_)), key=lambda d: d["rank"])[:25]
    elif hasattr(s, "estimator_") and hasattr(s.estimator_, "feature_importances_"):
        ranking = sorted(({"feature": names[i], "score": _r(v)} for i, v in enumerate(s.estimator_.feature_importances_)), key=lambda d: -(d["score"] or 0))[:25]
    a, b = float(all_scores.mean()), float(sel_scores.mean())
    if b >= a - 0.005:
        verdict = f"Keeping {k} of {n_feat} columns scores {b:.3f} against {a:.3f} with all of them. The smaller model is just as good, so the dropped columns were not needed."
    else:
        verdict = f"Keeping only {k} of {n_feat} columns drops the score from {a:.3f} to {b:.3f}. The dropped columns still carry signal; keep more of them."
    return {
        "task": task, "target": target, "method": method, "model": key, "model_name": reg[key]["name"], "scoring": scoring, "rows_used": len(X), "subsampled": sub,
        "k": k, "n_features": n_feat, "selected": [n for n, m_ in zip(names, mask) if m_], "dropped": [n for n, m_ in zip(names, mask) if not m_][:60],
        "ranking": ranking, "cv_all": _r(a), "cv_all_std": _r(all_scores.std()), "cv_selected": _r(b), "cv_selected_std": _r(sel_scores.std()),
        "verdict": verdict, "preprocessing": P["prep_info"], "balance": P.get("balance"), "log_target": P.get("log_target"), "selector_code": how,
        "narration": [
            {"step": "Method", "text": why, "code": how},
            {"step": "Inside the pipeline", "text": TEXT["select"], "code": "Pipeline([('prep', preprocessor), ('select', selector), ('model', model)])"},
            {"step": "Result", "text": verdict, "code": "cross_val_score(pipe_selected, X, y, cv=cv)"},
        ],
    }


# ---- calibration --------------------------------------------------------------

def _bins(conf, correct, n_bins: int = 10) -> tuple[list, float]:
    """Sort rows into equal-size confidence bins; compare stated confidence with how often it was right."""
    order = np.argsort(conf)
    pts, ece = [], 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk) == 0:
            continue
        c, a = float(conf[chunk].mean()), float(correct[chunk].mean())
        pts.append({"confidence": _r(c), "actual": _r(a), "rows": int(len(chunk))})
        ece += abs(c - a) * len(chunk) / len(conf)
    return pts, ece


def calibration(df, params: dict, progress, deadline) -> dict:
    task, target = params["task"], params["target"]
    if task != "classification":
        raise ValueError("Calibration is about predicted probabilities, so it only applies to classification.")
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, sub = preprocess.subsample(P["X"], P["y"], config.TUNE_ROWS, True)
    est, name, _ = build_estimator(task, params.get("spec") or {"kind": "single", "model": params.get("model") or "rf"}, P)
    if not hasattr(est, "predict_proba"):
        raise ValueError(f"{name} does not give probabilities, so there is nothing to calibrate.")
    folds = min(config.CV_FOLDS, int(params.get("folds") or config.CV_FOLDS))
    cv = _cv(task, folds)
    yv = np.asarray(y)
    binary = len(P["classes"]) == 2
    out_curves, out_scores = {}, {}
    variants = [("as trained", est), ("sigmoid (Platt)", CalibratedClassifierCV(clone(est), method="sigmoid", cv=3)),
                ("isotonic", CalibratedClassifierCV(clone(est), method="isotonic", cv=3))]
    for i, (label, e) in enumerate(variants):
        progress(f"Probabilities: {label}", 0.1 + 0.28 * i)
        proba = cross_val_predict(make_pipe(clone(P["prep"]), e, P), X, y, cv=cv, method="predict_proba", n_jobs=1)
        if binary:
            conf, correct = proba[:, 1], (yv == 1).astype(int)
            brier = float(np.mean((conf - correct) ** 2))
        else:
            conf, correct = proba.max(axis=1), (proba.argmax(axis=1) == yv).astype(int)
            onehot = np.eye(proba.shape[1])[yv]
            brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        pts, ece = _bins(conf, correct)
        out_curves[label] = pts
        out_scores[label] = {"brier": _r(brier), "ece": _r(ece)}
    raw = out_scores["as trained"]
    best = min(out_scores, key=lambda k: out_scores[k]["ece"] or 1)
    over = sum(1 for p in out_curves["as trained"] if (p["confidence"] or 0) > (p["actual"] or 0) + 0.03)
    under = sum(1 for p in out_curves["as trained"] if (p["actual"] or 0) > (p["confidence"] or 0) + 0.03)
    if raw["ece"] < 0.03:
        verdict = [f"{name} is already well calibrated: its stated confidence is off by {raw['ece']*100:.1f} points on average."]
    else:
        verdict = [f"{name} is {'overconfident' if over >= under else 'underconfident'}: its stated confidence is off by {raw['ece']*100:.1f} points on average."]
    if best != "as trained":
        verdict.append(f"{best.capitalize()} calibration brings that to {out_scores[best]['ece']*100:.1f} points (Brier score {raw['brier']:.3f} to {out_scores[best]['brier']:.3f}; lower is better).")
    else:
        verdict.append("Neither calibration method improved it, so keep the model as it is.")
    return {
        "task": task, "target": target, "model_name": name, "rows_used": len(X), "subsampled": sub, "binary": binary,
        "positive_class": P["classes"][1] if binary else None, "curves": out_curves, "scores": out_scores, "best": best, "verdict": verdict,
        "preprocessing": P["prep_info"], "balance": P.get("balance"),
        "narration": [
            {"step": "What calibration means", "text": TEXT["calibration"], "code": "calibration_curve(y_true, proba, n_bins=10)"},
            {"step": "How it was measured", "text": ("Probabilities of '" + P["classes"][1] + "'" if binary else "The confidence in the predicted class") + f" were collected with {folds}-part cross-validation, so every row is scored by a model that never saw it. Rows are put into 10 equal-size bins by confidence, and each bin's average confidence is compared with how often it was right. ECE is the average gap. The Brier score is the squared error of the probabilities.", "code": "cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')"},
            {"step": "Fixes compared", "text": "Sigmoid (Platt) fits an S-curve that maps the model's scores to probabilities; it works with little data. Isotonic fits any rising step shape; it is more flexible but needs more rows. " + " ".join(verdict), "code": "CalibratedClassifierCV(model, method='sigmoid', cv=3)"},
        ],
    }
