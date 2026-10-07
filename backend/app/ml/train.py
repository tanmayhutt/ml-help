"""Supervised jobs: the model race, single training, ensembles, tuning, and learning curves.
Every function receives a `progress` callback and a `deadline` (epoch seconds) and stops
cleanly when the budget runs out, so a heavy dataset never runs away."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .. import config
from . import catalog, evaluate, preprocess

Progress = Callable[[str, float], None]


def prepare(df: pd.DataFrame, target: str, task: str, prep_options: dict | None) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    data = df[df[target].notna()].copy()
    if preprocess.normalize(prep_options).get("drop_duplicates"):
        data = data.drop_duplicates()
    X = data.drop(columns=[target])
    y = data[target]
    if len(data) < 6:
        raise ValueError(f"Only {len(data)} usable rows. At least 6 are needed, and 50 or more to get meaningful scores.")
    if task == "classification":
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        classes = [str(c) for c in le.classes_]
        counts = y.astype(str).value_counts()
        if len(counts) < 2:
            raise ValueError(f"'{target}' has only one value ('{counts.index[0]}'). There is nothing to predict; the answer column needs at least two different values.")
        if counts.min() < 2:
            raise ValueError(f"Class '{counts.idxmin()}' has only one row. Every class needs at least 2 rows.")
    else:
        y_enc = pd.to_numeric(y, errors="coerce")
        if y_enc.notna().sum() == 0:
            raise ValueError(f"'{target}' is text, not numbers. Pick 'category' instead of 'number' for this column.")
        if y_enc.isna().any():
            keep = y_enc.notna()
            X, y_enc = X[keep], y_enc[keep]
        y_enc = y_enc.astype(float)
        classes = None
    try:
        ct, prep_info = preprocess.build(X, prep_options)
    except ValueError as e:
        raise ValueError("Every other column was dropped as an ID, a constant, or empty, so there is nothing to learn from. Add columns with real information about each row.") from e
    opts = preprocess.normalize(prep_options)
    out = {"X": X, "y": y_enc, "classes": classes, "prep": ct, "prep_info": prep_info, "balance": False, "log_target": False}
    if task == "classification":
        share = y_enc.value_counts(normalize=True)
        out["minority_share"] = round(float(share.min()), 4)
        want = prep_options.get("balance", "auto") if prep_options else "auto"
        out["balance"] = bool(want is True or want == "on" or (want == "auto" and share.min() < 0.35))
    else:
        want = prep_options.get("log_target", "auto") if prep_options else "auto"
        skew = float(y_enc.skew()) if len(y_enc) > 10 else 0.0
        out["target_skew"] = round(skew, 3)
        out["log_target"] = bool(want is True or want == "on" or (want == "auto" and y_enc.min() >= 0 and abs(skew) >= 1.0))
    return out


def make_model(task: str, key: str, P: dict):
    """Model from the shelf with the refinements that apply: balanced class weights, log-transformed target."""
    reg = catalog.registry(task)
    est = reg[key]["make"]()
    if P.get("balance"):
        catalog.set_balanced(est, key)
    if P.get("log_target"):
        est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
    return est


def _cv(task: str, n_splits: int):
    if task == "classification":
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    return KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def _scoring(task: str, metric: str | None) -> str:
    if task == "classification":
        return metric if metric in {"accuracy", "f1_macro", "balanced_accuracy", "roc_auc_ovr"} else "accuracy"
    return metric if metric in {"r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"} else "r2"


def _split(X, y, task, test_size: float):
    strat = y if task == "classification" else None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=strat)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)


def leaderboard(df, params: dict, progress: Progress, deadline: float) -> dict:
    task, target = params["task"], params["target"]
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, _ = preprocess.subsample(P["X"], P["y"], config.LEADERBOARD_ROWS, task == "classification")
    subsampled = len(X) < len(P["X"])
    reg = catalog.registry(task)
    wanted = params.get("models") or list(reg.keys())
    scoring = _scoring(task, params.get("metric"))
    folds = min(config.CV_FOLDS, int(params.get("folds") or config.CV_FOLDS))
    cv = _cv(task, folds)
    rows: list[dict] = []
    narration = []
    if P.get("balance"):
        narration.append({"step": "Class balance", "text": f"The smallest class is only {P['minority_share']*100:.0f}% of the rows. Models that support it are told to weigh the rare class more (class_weight='balanced'), so they cannot score well just by always guessing the common class.", "code": "LogisticRegression(class_weight='balanced')  # same for SVM, trees, forests"})
    if P.get("log_target"):
        narration.append({"step": "Answer column", "text": f"The answer column has a long tail (skew {P['target_skew']}). Every model is trained to predict log(1 + value) and its predictions are converted back, which usually cuts the error on big values a lot.", "code": "TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)"})
    narration += [
        {"step": "Data", "text": f"Using {len(X)} rows and {X.shape[1]} columns." + (f" Only {len(X)} of your {len(P['X'])} rows were used here so the test stays fast. Train the winner afterwards to use all rows." if subsampled else ""), "code": f"X = df.drop(columns=['{target}']); y = df['{target}']"},
        {"step": "Validation", "text": f"The rows are cut into {folds} parts. Each model is trained {folds} times, each time hiding a different part and scoring on it. The {folds} scores are averaged. This is called cross-validation and it stops one lucky split from fooling you.", "code": f"cross_val_score(pipe, X, y, cv={folds}, scoring='{scoring}')"},
    ]
    total = len(wanted)
    for i, key in enumerate(wanted):
        if key not in reg:
            continue
        m = reg[key]
        progress(f"Racing {m['name']}", i / max(total, 1))
        if time.time() > deadline - 5:
            rows.append({"key": key, "name": m["name"], "family": m["family"], "status": "skipped", "note": "Time budget exhausted."})
            continue
        if m.get("slow") and len(X) > config.SLOW_MODEL_ROWS and not params.get("include_slow"):
            rows.append({"key": key, "name": m["name"], "family": m["family"], "status": "skipped", "note": f"Skipped: this model scales badly above {config.SLOW_MODEL_ROWS} rows. Tick 'include slow models' to force it."})
            continue
        pipe = Pipeline([("prep", clone(P["prep"])), ("model", make_model(task, key, P))])
        t0 = time.time()
        try:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            rows.append({
                "key": key, "name": m["name"], "family": m["family"], "status": "ok",
                "mean": round(float(scores.mean()), 5), "std": round(float(scores.std()), 5),
                "folds": [round(float(s), 5) for s in scores], "seconds": round(time.time() - t0, 2),
            })
        except Exception as e:  # noqa: BLE001
            rows.append({"key": key, "name": m["name"], "family": m["family"], "status": "error", "note": str(e)[:200]})
    ok = [r for r in rows if r["status"] == "ok"]
    ok.sort(key=lambda r: r["mean"], reverse=True)
    ranked = ok + [r for r in rows if r["status"] != "ok"]
    summary = _leaderboard_summary(ranked, task, scoring)
    return {
        "task": task, "target": target, "scoring": scoring, "rows_used": len(X), "rows_total": len(P["X"]),
        "subsampled": subsampled, "folds": folds, "leaderboard": ranked,
        "preprocessing": P["prep_info"], "narration": narration, "summary": summary,
    }


def _leaderboard_summary(ranked: list[dict], task: str, scoring: str) -> list[str]:
    ok = [r for r in ranked if r["status"] == "ok"]
    if not ok:
        return ["No model finished. Check the error notes."]
    best, worst = ok[0], ok[-1]
    out = [f"{best['name']} scored best: {best['mean']:.3f}. Its score moved by about {best['std']:.3f} between test rounds, so treat small differences between models as a tie."]
    spread = best["mean"] - worst["mean"]
    if spread < 0.02:
        out.append("All the models score almost the same. The model choice does not matter much here; better columns would help more.")
    else:
        out.append(f"The best and worst models are {spread:.3f} apart, so the choice of model matters for this data.")
    ens = [r for r in ok if r["family"] in ("bagging", "boosting")]
    single = [r for r in ok if r["family"] not in ("bagging", "boosting")]
    if ens and single and max(r["mean"] for r in ens) > max(r["mean"] for r in single):
        out.append("Models that combine many trees did best. That is common for tables: many trees voting together make fewer mistakes than one model.")
    elif ens and single:
        out.append(f"A single simple model ({single[0]['name']}) did as well as the combined ones. The pattern in this data is probably simple.")
    return out


def build_estimator(task: str, spec: dict, P: dict | None = None) -> tuple[Any, str, list[dict]]:
    """spec: {"kind": "single"|"voting"|"stacking"|"bagging", "model": key, "members": [keys], "final": key, "params": {...}}"""
    reg = catalog.registry(task)
    P = P or {}
    kind = spec.get("kind", "single")
    narr: list[dict] = []
    if kind == "single":
        key = spec.get("model")
        if key not in reg:
            raise ValueError(f"Unknown model '{key}'.")
        est = reg[key]["make"]()
        est.set_params(**_clean_params(spec.get("params"), est))
        if P.get("balance"):
            catalog.set_balanced(est, key)
        if P.get("log_target"):
            est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
        inner = est.regressor if isinstance(est, TransformedTargetRegressor) else est
        narr.append({"step": "Model", "text": f"{reg[key]['name']}: {reg[key]['explain']}", "code": f"{inner.__class__.__name__}({_fmt_params(inner, reg[key])})"})
        return est, reg[key]["name"], narr
    members = [k for k in (spec.get("members") or []) if k in reg]
    if kind in ("voting", "stacking") and len(members) < 2:
        raise ValueError("Pick at least two member models for an ensemble.")
    if kind == "voting":
        ests = [(k, make_model(task, k, P) if task == "classification" else reg[k]["make"]()) for k in members]
        if task == "classification":
            voting = spec.get("voting", "soft")
            if voting == "soft":
                # all members must support predict_proba
                for k, e in ests:
                    if k == "svm":
                        e.set_params(probability=True)
            est = VotingClassifier(ests, voting=voting, n_jobs=1)
            name = f"{voting.title()} Voting ({', '.join(reg[k]['name'] for k in members)})"
            narr.append({"step": "Ensemble", "text": f"{catalog.ENSEMBLE_TEXT['voting']} Members: {', '.join(reg[k]['name'] for k in members)}.", "code": f"VotingClassifier([...], voting='{voting}')"})
        else:
            est = VotingRegressor(ests, n_jobs=1)
            if P.get("log_target"):
                est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
            name = f"Voting ({', '.join(reg[k]['name'] for k in members)})"
            narr.append({"step": "Ensemble", "text": f"Voting regressor: averages the predictions of {', '.join(reg[k]['name'] for k in members)}.", "code": "VotingRegressor([...])"})
        return est, name, narr
    if kind == "stacking":
        ests = [(k, make_model(task, k, P) if task == "classification" else reg[k]["make"]()) for k in members]
        final_key = spec.get("final")
        if task == "classification":
            final = reg[final_key]["make"]() if final_key in reg else LogisticRegression(max_iter=1000)
            est = StackingClassifier(ests, final_estimator=final, cv=3, n_jobs=1, stack_method="auto")
        else:
            final = reg[final_key]["make"]() if final_key in reg else Ridge()
            est = StackingRegressor(ests, final_estimator=final, cv=3, n_jobs=1)
            if P.get("log_target"):
                est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1)
        name = f"Stacking ({', '.join(reg[k]['name'] for k in members)}) -> {final.__class__.__name__}"
        narr.append({"step": "Ensemble", "text": f"{catalog.ENSEMBLE_TEXT['stacking']} Base: {', '.join(reg[k]['name'] for k in members)}. Meta-learner: {final.__class__.__name__}.", "code": f"Stacking{'Classifier' if task == 'classification' else 'Regressor'}([...], final_estimator={final.__class__.__name__}(), cv=3)"})
        return est, name, narr
    if kind == "bagging":
        base_key = spec.get("model") or "tree"
        if base_key not in reg:
            raise ValueError(f"Unknown base model '{base_key}'.")
        n = int(min(max(spec.get("n_estimators", 25), 2), 100))
        base = reg[base_key]["make"]()
        cls = BaggingClassifier if task == "classification" else BaggingRegressor
        est = cls(estimator=base, n_estimators=n, n_jobs=1, random_state=config.RANDOM_STATE)
        name = f"Bagging x{n} of {reg[base_key]['name']}"
        narr.append({"step": "Ensemble", "text": f"{catalog.ENSEMBLE_TEXT['bagging']} Here: {n} copies of {reg[base_key]['name']}, each on a bootstrap sample.", "code": f"{cls.__name__}(estimator={base.__class__.__name__}(), n_estimators={n})"})
        return est, name, narr
    raise ValueError(f"Unknown ensemble kind '{kind}'.")


def _clean_params(params: dict | None, est) -> dict:
    if not params:
        return {}
    valid = est.get_params()
    out = {}
    for k, v in params.items():
        if k in valid and v is not None and v != "":
            out[k] = v
    return out


def _fmt_params(est, meta: dict) -> str:
    p = est.get_params()
    keys = [k.replace("model__", "") for k in meta["space"].keys()]
    return ", ".join(f"{k}={p[k]!r}" for k in keys if k in p)


def train(df, params: dict, progress: Progress, deadline: float) -> dict:
    task, target = params["task"], params["target"]
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, _ = preprocess.subsample(P["X"], P["y"], config.TRAIN_ROWS, task == "classification")
    test_size = float(min(max(params.get("test_size", 0.2), 0.1), 0.5))
    Xtr, Xte, ytr, yte = _split(X, y, task, test_size)
    est, name, narr = build_estimator(task, params.get("spec") or {"kind": "single", "model": params.get("model")}, P)
    pipe = Pipeline([("prep", P["prep"]), ("model", est)])
    narration = [
        {"step": "Split", "text": f"{len(Xtr)} rows are used to teach the model. {len(Xte)} rows ({int(test_size*100)}%) are hidden and used only to test it afterwards, so the score shows how it does on rows it has never seen.", "code": f"train_test_split(X, y, test_size={test_size}, random_state=42{', stratify=y' if task == 'classification' else ''})"},
    ] + narr
    progress(f"Training {name}", 0.2)
    t0 = time.time()
    pipe.fit(Xtr, ytr)
    fit_s = time.time() - t0
    progress("Evaluating", 0.6)
    pred = pipe.predict(Xte)
    train_pred = pipe.predict(Xtr)
    names = preprocess.feature_names(pipe.named_steps["prep"])
    threshold = None
    if task == "classification":
        proba = pipe.predict_proba(Xte) if hasattr(pipe, "predict_proba") else None
        threshold = params.get("threshold")
        if threshold and proba is not None and proba.shape[1] == 2:
            pred = (proba[:, 1] >= float(threshold)).astype(int)
        ev = evaluate.classification(yte, pred, proba, P["classes"])
        train_score = float(evaluate.M.accuracy_score(ytr, train_pred))
    else:
        ev = evaluate.regression(yte, pred, n_features=len(names) or X.shape[1])
        train_score = float(evaluate.M.r2_score(ytr, train_pred))
    test_score = ev["metrics"][ev["primary"]]
    ev["train_score"] = round(train_score, 5)
    ev["fit_diagnosis"] = _fit_diagnosis(train_score, test_score)
    progress("Measuring feature importance", 0.8)
    if time.time() < deadline - 10:
        ev["importance"] = evaluate.feature_importance(pipe, names, Xte, yte, task, config.PERM_IMPORTANCE_ROWS)
    narration.append({"step": "Fit", "text": f"Training took {fit_s:.2f} seconds. Score on the training rows: {train_score:.4f}. Score on the hidden test rows: {test_score:.4f}. {ev['fit_diagnosis']['text']}", "code": "pipe.fit(X_train, y_train); pipe.score(X_test, y_test)"})
    return {
        "task": task, "target": target, "model_name": name, "rows_used": len(X), "rows_total": len(P["X"]),
        "test_size": test_size, "evaluation": ev, "preprocessing": P["prep_info"], "narration": narration,
        "feature_names": names[:200], "features": list(P["X"].columns), "classes": P["classes"], "threshold": threshold,
        "balance": P.get("balance"), "log_target": P.get("log_target"),
        "_pipeline": pipe,
    }


def _fit_diagnosis(train_score: float, test_score: float | None) -> dict:
    if test_score is None:
        return {"label": "unknown", "text": ""}
    gap = train_score - test_score
    if train_score < 0.6 and test_score < 0.6:
        return {"label": "underfitting", "text": "Scores are low on both training and test rows. The model is too simple, or the columns do not contain enough information."}
    if gap > 0.15:
        return {"label": "overfitting", "text": "The score on training rows is far above the score on test rows: the model memorised instead of learning. Use a simpler setting, more rows, or a combined model."}
    if gap > 0.05:
        return {"label": "slight overfitting", "text": "A small gap between training and test rows. Normal for tree models."}
    return {"label": "good fit", "text": "Training and test scores agree. The model should work on new rows."}


def tune(df, params: dict, progress: Progress, deadline: float) -> dict:
    task, target, key = params["task"], params["target"], params.get("model")
    reg = catalog.registry(task)
    if key not in reg:
        raise ValueError(f"Unknown model '{key}'.")
    m = reg[key]
    if not m["space"]:
        raise ValueError(f"{m['name']} has no hyperparameters worth tuning.")
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, sub = preprocess.subsample(P["X"], P["y"], config.TUNE_ROWS, task == "classification")
    n_iter = int(min(max(params.get("n_iter", 12), 2), config.TUNE_MAX_ITER))
    folds = min(config.CV_FOLDS, int(params.get("folds") or config.CV_FOLDS))
    scoring = _scoring(task, params.get("metric"))
    pipe = Pipeline([("prep", P["prep"]), ("model", make_model(task, key, P))])
    space = {("model__regressor__" + k.split("model__", 1)[1] if P.get("log_target") else k): v for k, v in m["space"].items()}
    progress(f"Searching {n_iter} configurations of {m['name']}", 0.1)
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=_cv(task, folds), scoring=scoring, n_jobs=1, random_state=config.RANDOM_STATE, refit=False, error_score=np.nan)
    search.fit(X, y)
    res = search.cv_results_
    trials = []
    for i in range(len(res["mean_test_score"])):
        p = {k.replace("model__regressor__", "").replace("model__", ""): _jsonable(v) for k, v in res["params"][i].items()}
        trials.append({"params": p, "mean": _r(res["mean_test_score"][i]), "std": _r(res["std_test_score"][i]), "seconds": _r(res["mean_fit_time"][i])})
    trials.sort(key=lambda t: (t["mean"] is None, -(t["mean"] or 0)))
    base = cross_val_score(Pipeline([("prep", clone(P["prep"])), ("model", make_model(task, key, P))]), X, y, cv=_cv(task, folds), scoring=scoring, n_jobs=1)
    best = trials[0]
    narration = [
        {"step": "Search", "text": f"{n_iter} random combinations of settings for {m['name']} were tried. Each one was scored with {folds}-part cross-validation, so {n_iter * folds} models were trained in total.", "code": f"RandomizedSearchCV(pipe, space, n_iter={n_iter}, cv={folds}, scoring='{scoring}')"},
        {"step": "Result", "text": f"The default settings score {base.mean():.4f}. The best settings found score {best['mean']:.4f}: {best['params']}. " + ("That is a real improvement." if (best["mean"] or 0) - base.mean() > 0.005 else "That is almost no change. The defaults were already good for this data."), "code": "search.best_params_"},
    ]
    return {
        "task": task, "target": target, "model": key, "model_name": m["name"], "scoring": scoring, "rows_used": len(X), "subsampled": sub,
        "baseline": _r(base.mean()), "trials": trials, "best_params": best["params"], "param_help": m["params"], "narration": narration,
        "preprocessing": P["prep_info"],
    }


def curve(df, params: dict, progress: Progress, deadline: float) -> dict:
    """Learning curve: score versus training size. Five points, three folds, capped rows."""
    task, target = params["task"], params["target"]
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, _ = preprocess.subsample(P["X"], P["y"], config.TUNE_ROWS, task == "classification")
    est, name, _ = build_estimator(task, params.get("spec") or {"kind": "single", "model": params.get("model")}, P)
    pipe = Pipeline([("prep", P["prep"]), ("model", est)])
    scoring = _scoring(task, params.get("metric"))
    progress(f"Learning curve for {name}", 0.2)
    sizes, tr, te = learning_curve(pipe, X, y, cv=_cv(task, 3), scoring=scoring, train_sizes=np.linspace(0.15, 1.0, 5), n_jobs=1, random_state=config.RANDOM_STATE, shuffle=True)
    pts = [{"n": int(s), "train": _r(a.mean()), "test": _r(b.mean()), "test_std": _r(b.std())} for s, a, b in zip(sizes, tr, te)]
    last = pts[-1]
    if last["train"] - last["test"] > 0.1 and last["test"] < pts[-2]["test"] + 0.01:
        verdict = "The model is memorising: it scores high on training rows but much lower on new rows. More rows or a simpler model would help."
    elif last["test"] - pts[0]["test"] > 0.03:
        verdict = "Yes, more data would help: the score on new rows is still rising."
    elif last["train"] < 0.7 and task == "classification":
        verdict = "The model is too simple: even on its own training rows it scores low. Try a more flexible model."
    else:
        verdict = "More data would not help much: the two lines have flattened and met. Better columns or a different model might."
    return {"model_name": name, "scoring": scoring, "points": pts, "verdict": verdict, "preprocessing": P["prep_info"],
            "narration": [{"step": "Learning curve", "text": "The model is trained again and again on bigger and bigger slices of your rows. If the score on new rows keeps rising, more data would help. If the training score is high but the new-row score stays low, the model is memorising instead of learning.", "code": "learning_curve(pipe, X, y, cv=3, train_sizes=np.linspace(0.15, 1, 5))"}]}


def _r(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) else round(v, 5)
    except (TypeError, ValueError):
        return None


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6)
    if isinstance(v, tuple):
        return list(v)
    return v


def auto(df, params: dict, progress: Progress, deadline: float) -> dict:
    """One click: race every model, build ensembles from the best ones, train the overall winner, explain the choice."""
    task, target = params["task"], params["target"]
    n_rows = int(df[target].notna().sum())
    params = dict(params)
    params.setdefault("include_slow", n_rows <= config.LEADERBOARD_ROWS)
    params.pop("models", None)
    progress("Checking every model", 0.05)
    race = leaderboard(df, params, lambda m, f: progress(m, 0.05 + f * 0.5), deadline)
    ok = [r for r in race["leaderboard"] if r["status"] == "ok"]
    if not ok:
        raise ValueError("No model could be trained on this column.")
    reg = catalog.registry(task)
    scoring = race["scoring"]
    folds = race["folds"]
    P = prepare(df, target, task, params.get("preprocess"))
    X, y, _ = preprocess.subsample(P["X"], P["y"], config.LEADERBOARD_ROWS, task == "classification")
    cv = _cv(task, folds)
    top = [r["key"] for r in ok[:3]]
    ensembles: list[dict] = []
    specs = []
    if len(top) >= 2:
        specs.append(("Soft voting of top 3" if task == "classification" else "Voting of top 3", {"kind": "voting", "members": top, "voting": "soft"}))
        specs.append(("Stacking of top 3", {"kind": "stacking", "members": top, "final": "logreg" if task == "classification" else "ridge"}))
    for i, (label, spec) in enumerate(specs):
        if time.time() > deadline - 20:
            ensembles.append({"key": f"ens{i}", "name": label, "family": "ensemble", "status": "skipped", "note": "Time budget exhausted.", "spec": spec})
            continue
        progress(f"Trying {label.lower()}", 0.55 + i * 0.08)
        try:
            est, name, _ = build_estimator(task, spec, P)
            pipe = Pipeline([("prep", clone(P["prep"])), ("model", est)])
            t0 = time.time()
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            ensembles.append({"key": f"ens{i}", "name": label, "family": "ensemble", "status": "ok", "mean": round(float(scores.mean()), 5), "std": round(float(scores.std()), 5),
                              "folds": [round(float(v), 5) for v in scores], "seconds": round(time.time() - t0, 2), "spec": spec, "members": [reg[k]["name"] for k in top]})
        except Exception as e:  # noqa: BLE001
            ensembles.append({"key": f"ens{i}", "name": label, "family": "ensemble", "status": "error", "note": str(e)[:200], "spec": spec})
    everything = ok + [e for e in ensembles if e["status"] == "ok"]
    everything.sort(key=lambda r: r["mean"], reverse=True)
    best = everything[0]
    runner_up = everything[1] if len(everything) > 1 else None
    best_spec = best.get("spec") or {"kind": "single", "model": best["key"]}
    refinements: list[dict] = []
    opts = preprocess.normalize(params.get("preprocess"))
    cv_params = {**params}

    def cv_score(spec, prep_opts):
        Pq = prepare(df, target, task, prep_opts)
        Xq, yq, _ = preprocess.subsample(Pq["X"], Pq["y"], config.LEADERBOARD_ROWS, task == "classification")
        est_q, _, _ = build_estimator(task, spec, Pq)
        return float(cross_val_score(Pipeline([("prep", Pq["prep"]), ("model", est_q)]), Xq, yq, cv=_cv(task, folds), scoring=scoring, n_jobs=1).mean())

    # 1. Feature selection: drop columns the winner does not use, keep the change only if the score holds.
    if params.get("feature_select", True) and time.time() < deadline - 60 and len(P["X"].columns) >= 4:
        progress("Checking which columns the winner really needs", 0.72)
        try:
            est_w, _, _ = build_estimator(task, best_spec, P)
            pipe_w = Pipeline([("prep", clone(P["prep"])), ("model", est_w)])
            Xtr, Xte, ytr, yte = _split(X, y, task, 0.25)
            pipe_w.fit(Xtr, ytr)
            n = min(len(Xte), config.PERM_IMPORTANCE_ROWS)
            idx = np.random.RandomState(0).choice(len(Xte), n, replace=False)
            imp = permutation_importance(pipe_w, Xte.iloc[idx], np.asarray(yte)[idx], n_repeats=3, random_state=0, scoring=scoring, n_jobs=1)
            useless = [c for c, m_, s_ in zip(Xte.columns, imp.importances_mean, imp.importances_std) if m_ <= 0.0005]
            if useless and len(useless) < len(Xte.columns):
                before = best["mean"]
                new_opts = {**opts, "drop_columns": list(set(opts["drop_columns"]) | set(useless))}
                after = cv_score(best_spec, new_opts)
                keep = after >= before - 0.002
                refinements.append({"step": "Drop columns the model does not use", "applied": keep, "before": round(before, 4), "after": round(after, 4),
                                    "text": f"Shuffling {', '.join(useless)} did not hurt the winner at all, so those columns carry no usable signal. " + (f"Without them the score went from {before:.3f} to {after:.3f}, so they are dropped: a simpler model that is just as good." if keep else f"Without them the score dropped from {before:.3f} to {after:.3f}, so they are kept."),
                                    "code": f"X = X.drop(columns={useless!r})"})
                if keep:
                    opts = new_opts; cv_params["preprocess"] = opts; best["mean"] = round(after, 5)
            else:
                refinements.append({"step": "Drop columns the model does not use", "applied": False, "text": "Every column changed the winner's score when shuffled, so all of them stay.", "code": "permutation_importance(pipe, X_test, y_test, n_repeats=3)"})
        except Exception as e:  # noqa: BLE001
            refinements.append({"step": "Drop columns the model does not use", "applied": False, "text": f"Skipped: {type(e).__name__}.", "code": ""})

    # 2. Tune the winner's settings when it is a single model with a search space.
    if params.get("tune_winner", True) and best_spec.get("kind") == "single" and reg.get(best_spec.get("model"), {}).get("space") and time.time() < deadline - 50:
        progress(f"Tuning {best['name']}", 0.8)
        try:
            tr = tune(df, {**cv_params, "model": best_spec["model"], "n_iter": 8}, lambda m, f: None, deadline - 15)
            if tr["trials"] and tr["trials"][0]["mean"] is not None and tr["trials"][0]["mean"] > tr["baseline"] + 0.002:
                best_spec = {**best_spec, "params": tr["best_params"]}
                refinements.append({"step": "Tune the winner's settings", "applied": True, "before": tr["baseline"], "after": tr["trials"][0]["mean"], "text": f"8 random settings were tried. {tr['best_params']} beat the defaults: {tr['baseline']:.3f} to {tr['trials'][0]['mean']:.3f}.", "code": f"RandomizedSearchCV(pipe, space, n_iter=8, cv={folds})"})
                best["mean"] = tr["trials"][0]["mean"]; best["name"] = best["name"] + " (tuned)"
            else:
                refinements.append({"step": "Tune the winner's settings", "applied": False, "before": tr["baseline"], "after": (tr["trials"][0]["mean"] if tr["trials"] else None), "text": "8 random settings were tried; none beat the defaults by a meaningful amount, so the defaults stay.", "code": f"RandomizedSearchCV(pipe, space, n_iter=8, cv={folds})"})
        except Exception as e:  # noqa: BLE001
            refinements.append({"step": "Tune the winner's settings", "applied": False, "text": f"Skipped: {type(e).__name__}.", "code": ""})

    # 3. Decision threshold for imbalanced two-class problems.
    threshold = None
    if task == "classification" and P.get("balance") and P["classes"] and len(P["classes"]) == 2 and time.time() < deadline - 30:
        progress("Choosing the decision threshold", 0.86)
        try:
            Pq = prepare(df, target, task, opts)
            Xq, yq, _ = preprocess.subsample(Pq["X"], Pq["y"], config.LEADERBOARD_ROWS, True)
            est_q, _, _ = build_estimator(task, best_spec, Pq)
            pipe_q = Pipeline([("prep", Pq["prep"]), ("model", est_q)])
            if hasattr(est_q, "predict_proba") or hasattr(est_q, "decision_function"):
                proba = cross_val_predict(pipe_q, Xq, yq, cv=_cv(task, folds), method="predict_proba", n_jobs=1)[:, 1]
                yq_arr = np.asarray(yq)
                base_ba = float(evaluate.M.balanced_accuracy_score(yq_arr, (proba >= 0.5).astype(int)))
                best_t, best_ba = 0.5, base_ba
                for t in np.linspace(0.2, 0.8, 25):
                    ba = float(evaluate.M.balanced_accuracy_score(yq_arr, (proba >= t).astype(int)))
                    if ba > best_ba + 1e-9:
                        best_t, best_ba = float(t), ba
                if best_ba > base_ba + 0.005 and abs(best_t - 0.5) > 0.02:
                    threshold = round(best_t, 3)
                    refinements.append({"step": "Move the decision threshold", "applied": True, "before": round(base_ba, 4), "after": round(best_ba, 4), "text": f"By default a row is called '{P['classes'][1]}' when the model is at least 50% sure. Calling it at {threshold*100:.0f}% raises balanced accuracy from {base_ba:.3f} to {best_ba:.3f}, so the rare class is caught more often.", "code": f"pred = (pipe.predict_proba(X)[:, 1] >= {threshold})"})
                else:
                    refinements.append({"step": "Move the decision threshold", "applied": False, "text": "Shifting the 50% cut-off did not improve balanced accuracy, so it stays at 50%.", "code": "pred = (pipe.predict_proba(X)[:, 1] >= 0.5)"})
        except Exception as e:  # noqa: BLE001
            refinements.append({"step": "Move the decision threshold", "applied": False, "text": f"Skipped: {type(e).__name__}.", "code": ""})

    # Train the winner on a hold-out split for full charts and a saved model.
    progress(f"Training the winner: {best['name']}", 0.9)
    final = train(df, {**cv_params, "spec": best_spec, "threshold": threshold}, lambda m, f: None, deadline)
    # Plain-language verdict
    verdict = []
    noise = max(best.get("std", 0), (runner_up or {}).get("std", 0) or 0)
    if runner_up:
        gap = best["mean"] - runner_up["mean"]
        if gap <= noise:
            verdict.append(f"{best['name']} and {runner_up['name']} are a tie: the gap between them ({gap:.3f}) is smaller than the swing between test rounds ({noise:.3f}).")
        else:
            verdict.append(f"{best['name']} is clearly ahead of {runner_up['name']} by {gap:.3f}.")
    best_single = ok[0]
    best_ens = next((e for e in ensembles if e["status"] == "ok"), None)
    if best_ens and best["family"] == "ensemble":
        verdict.append(f"Combining the top models helped: {best['name']} beat the best single model ({best_single['name']}, {best_single['mean']:.3f}).")
    elif best_ens:
        verdict.append(f"Combining the top models did not beat {best_single['name']} on its own, so the simpler single model is recommended.")
    simple = next((r for r in ok if r["family"] in ("linear", "probabilistic", "tree") and best["mean"] - r["mean"] <= max(noise, 0.01)), None)
    if simple and simple["key"] != best.get("key"):
        verdict.append(f"{simple['name']} scores almost the same ({simple['mean']:.3f}) and is simpler and faster. Prefer it if you need to explain the model.")
    verdict.append(f"Final check on rows the model never saw: {final['evaluation']['primary']} {final['evaluation']['metrics'][final['evaluation']['primary']]:.3f}.")
    tried = [{"name": reg[r["key"]]["name"], "family": reg[r["key"]]["family"], "explain": reg[r["key"]]["explain"], "code": f"{reg[r['key']]['make']().__class__.__name__}({_fmt_params(reg[r['key']]['make'](), reg[r['key']])})", "mean": r.get("mean"), "status": r["status"]}
             for r in race["leaderboard"] if r["key"] in reg]
    narration = race["narration"] + [
        {"step": "The loop", "text": f"Every model is built the same way: a pipeline that first prepares the data and then fits the model. The code loops over the shelf, runs cross-validation for each one, and keeps the scores in a table. Nothing is special about any model in the loop; they all see exactly the same prepared rows and the same {folds} test rounds, which is what makes the ranking fair.", "code": "for name, model in models.items():\n    pipe = Pipeline([('prep', preprocessor), ('model', model)])\n    scores = cross_val_score(pipe, X, y, cv=cv, scoring='" + scoring + "')\n    results[name] = (scores.mean(), scores.std())"},
        {"step": "What an ensemble is", "text": "Cross-validation only measures a model. An ensemble is a new model made by combining several models. Voting: each member predicts, and the answers are averaged (for numbers) or the most confident class wins (for categories). Stacking: the members predict, and a small final model learns how much to trust each member. Bagging and boosting (Random Forest, Gradient Boosting) are ensembles of many trees built inside one model. Combining helps when the members make different mistakes.", "code": "VotingClassifier([('a', m1), ('b', m2), ('c', m3)], voting='soft')\nStackingClassifier([('a', m1), ('b', m2), ('c', m3)], final_estimator=LogisticRegression())"},
        {"step": "Ensembles", "text": f"The top three models ({', '.join(reg[k]['name'] for k in top)}) were combined two ways: voting, where they average their answers, and stacking, where a small final model learns how much to trust each one. Both were scored with the same {folds}-part cross-validation.", "code": f"VotingClassifier([...], voting='soft'); StackingClassifier([...], cv=3)" if task == "classification" else "VotingRegressor([...]); StackingRegressor([...], cv=3)"},
        {"step": "Winner", "text": f"{best['name']} had the best cross-validated {scoring} ({best['mean']:.4f}). It was then trained once more with 20% of the rows hidden, to produce the charts below and a saved model you can use.", "code": "pipe.fit(X_train, y_train)"},
    ]
    return {
        "task": task, "target": target, "scoring": scoring, "folds": folds, "rows_used": race["rows_used"], "rows_total": race["rows_total"], "subsampled": race["subsampled"],
        "leaderboard": everything + [r for r in race["leaderboard"] if r["status"] != "ok"] + [e for e in ensembles if e["status"] != "ok"],
        "best": {"name": best["name"], "mean": best["mean"], "std": best.get("std"), "family": best["family"], "spec": best_spec, "preprocess": opts},
        "verdict": verdict, "final": {k: v for k, v in final.items() if k not in ("narration", "preprocessing", "_pipeline", "code")}, "models_tried": tried,
        "refinements": refinements, "threshold": threshold, "balance": P.get("balance"), "log_target": P.get("log_target"),
        "features": final.get("features"), "classes": final.get("classes"), "model_name": final.get("model_name"), "test_size": final.get("test_size"),
        "model_id": None, "_pipeline": final.get("_pipeline"), "preprocessing": final.get("preprocessing") or race["preprocessing"], "narration": narration,
    }
