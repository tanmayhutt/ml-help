"""Supervised jobs: the model race, single training, ensembles, tuning, and learning curves.
Every function receives a `progress` callback and a `deadline` (epoch seconds) and stops
cleanly when the budget runs out, so a heavy dataset never runs away."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
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
    X = data.drop(columns=[target])
    y = data[target]
    if task == "classification":
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
        classes = [str(c) for c in le.classes_]
        counts = y.astype(str).value_counts()
        if counts.min() < 2:
            raise ValueError(f"Class '{counts.idxmin()}' has only one row. Every class needs at least 2 rows.")
    else:
        y_enc = pd.to_numeric(y, errors="coerce")
        if y_enc.isna().any():
            keep = y_enc.notna()
            X, y_enc = X[keep], y_enc[keep]
        y_enc = y_enc.astype(float)
        classes = None
    ct, prep_info = preprocess.build(X, prep_options)
    return {"X": X, "y": y_enc, "classes": classes, "prep": ct, "prep_info": prep_info}


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
    narration = [
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
        pipe = Pipeline([("prep", clone(P["prep"])), ("model", m["make"]())])
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


def build_estimator(task: str, spec: dict) -> tuple[Any, str, list[dict]]:
    """spec: {"kind": "single"|"voting"|"stacking"|"bagging", "model": key, "members": [keys], "final": key, "params": {...}}"""
    reg = catalog.registry(task)
    kind = spec.get("kind", "single")
    narr: list[dict] = []
    if kind == "single":
        key = spec.get("model")
        if key not in reg:
            raise ValueError(f"Unknown model '{key}'.")
        est = reg[key]["make"]()
        est.set_params(**_clean_params(spec.get("params"), est))
        narr.append({"step": "Model", "text": f"{reg[key]['name']}: {reg[key]['explain']}", "code": f"{est.__class__.__name__}({_fmt_params(est, reg[key])})"})
        return est, reg[key]["name"], narr
    members = [k for k in (spec.get("members") or []) if k in reg]
    if kind in ("voting", "stacking") and len(members) < 2:
        raise ValueError("Pick at least two member models for an ensemble.")
    if kind == "voting":
        ests = [(k, reg[k]["make"]()) for k in members]
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
            name = f"Voting ({', '.join(reg[k]['name'] for k in members)})"
            narr.append({"step": "Ensemble", "text": f"Voting regressor: averages the predictions of {', '.join(reg[k]['name'] for k in members)}.", "code": "VotingRegressor([...])"})
        return est, name, narr
    if kind == "stacking":
        ests = [(k, reg[k]["make"]()) for k in members]
        final_key = spec.get("final")
        if task == "classification":
            final = reg[final_key]["make"]() if final_key in reg else LogisticRegression(max_iter=1000)
            est = StackingClassifier(ests, final_estimator=final, cv=3, n_jobs=1, stack_method="auto")
        else:
            final = reg[final_key]["make"]() if final_key in reg else Ridge()
            est = StackingRegressor(ests, final_estimator=final, cv=3, n_jobs=1)
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
    est, name, narr = build_estimator(task, params.get("spec") or {"kind": "single", "model": params.get("model")})
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
    if task == "classification":
        proba = pipe.predict_proba(Xte) if hasattr(pipe, "predict_proba") else None
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
        "feature_names": names[:200], "features": list(P["X"].columns), "classes": P["classes"],
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
    pipe = Pipeline([("prep", P["prep"]), ("model", m["make"]())])
    progress(f"Searching {n_iter} configurations of {m['name']}", 0.1)
    search = RandomizedSearchCV(pipe, m["space"], n_iter=n_iter, cv=_cv(task, folds), scoring=scoring, n_jobs=1, random_state=config.RANDOM_STATE, refit=False, error_score=np.nan)
    search.fit(X, y)
    res = search.cv_results_
    trials = []
    for i in range(len(res["mean_test_score"])):
        p = {k.replace("model__", ""): _jsonable(v) for k, v in res["params"][i].items()}
        trials.append({"params": p, "mean": _r(res["mean_test_score"][i]), "std": _r(res["std_test_score"][i]), "seconds": _r(res["mean_fit_time"][i])})
    trials.sort(key=lambda t: (t["mean"] is None, -(t["mean"] or 0)))
    base = cross_val_score(Pipeline([("prep", clone(P["prep"])), ("model", m["make"]())]), X, y, cv=_cv(task, folds), scoring=scoring, n_jobs=1)
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
    est, name, _ = build_estimator(task, params.get("spec") or {"kind": "single", "model": params.get("model")})
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
        progress(f"Trying {label.lower()}", 0.6 + i * 0.12)
        try:
            est, name, _ = build_estimator(task, spec)
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
    # Train the winner on a hold-out split for full charts and a saved model.
    progress(f"Training the winner: {best['name']}", 0.85)
    final = train(df, {**params, "spec": best_spec}, lambda m, f: None, deadline)
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
    narration = race["narration"] + [
        {"step": "Ensembles", "text": f"The top three models ({', '.join(reg[k]['name'] for k in top)}) were combined two ways: voting, where they average their answers, and stacking, where a small final model learns how much to trust each one. Both were scored with the same {folds}-part cross-validation.", "code": f"VotingClassifier([...], voting='soft'); StackingClassifier([...], cv=3)" if task == "classification" else "VotingRegressor([...]); StackingRegressor([...], cv=3)"},
        {"step": "Winner", "text": f"{best['name']} had the best cross-validated {scoring} ({best['mean']:.4f}). It was then trained once more with 20% of the rows hidden, to produce the charts below and a saved model you can use.", "code": "pipe.fit(X_train, y_train)"},
    ]
    return {
        "task": task, "target": target, "scoring": scoring, "folds": folds, "rows_used": race["rows_used"], "rows_total": race["rows_total"], "subsampled": race["subsampled"],
        "leaderboard": everything + [r for r in race["leaderboard"] if r["status"] != "ok"] + [e for e in ensembles if e["status"] != "ok"],
        "best": {"name": best["name"], "mean": best["mean"], "std": best.get("std"), "family": best["family"], "spec": best_spec},
        "verdict": verdict, "final": {k: v for k, v in final.items() if k not in ("narration", "preprocessing", "_pipeline", "code")},
        "model_id": None, "_pipeline": final.get("_pipeline"), "preprocessing": race["preprocessing"], "narration": narration,
    }
