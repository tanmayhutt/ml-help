"""Metrics and the numbers behind every chart, each with a how-to-read note."""
from __future__ import annotations

import numpy as np
from sklearn import metrics as M
from sklearn.preprocessing import label_binarize


def _f(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else round(v, 6)
    except (TypeError, ValueError):
        return None


def classification(y_true, y_pred, proba, classes) -> dict:
    classes = [str(c) for c in classes]
    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "macro"
    labels = list(range(n_classes))
    out: dict = {
        "metrics": {
            "accuracy": _f(M.accuracy_score(y_true, y_pred)),
            "balanced_accuracy": _f(M.balanced_accuracy_score(y_true, y_pred)),
            "precision": _f(M.precision_score(y_true, y_pred, average=avg, zero_division=0)),
            "recall": _f(M.recall_score(y_true, y_pred, average=avg, zero_division=0)),
            "f1": _f(M.f1_score(y_true, y_pred, average=avg, zero_division=0)),
            "mcc": _f(M.matthews_corrcoef(y_true, y_pred)),
        },
        "primary": "accuracy",
        "classes": classes,
        "confusion": M.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": [],
    }
    rep = M.precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, c in enumerate(classes):
        out["per_class"].append({"class": c, "precision": _f(rep[0][i]), "recall": _f(rep[1][i]), "f1": _f(rep[2][i]), "support": int(rep[3][i])})
    if proba is not None:
        try:
            if n_classes == 2:
                p = proba[:, 1]
                out["metrics"]["roc_auc"] = _f(M.roc_auc_score(y_true, p))
                out["metrics"]["log_loss"] = _f(M.log_loss(y_true, proba, labels=labels))
                fpr, tpr, _ = M.roc_curve(y_true, p)
                prec, rec, _ = M.precision_recall_curve(y_true, p)
                out["roc"] = _thin(fpr, tpr)
                out["pr"] = _thin(rec, prec)
            else:
                yb = label_binarize(y_true, classes=labels)
                out["metrics"]["roc_auc"] = _f(M.roc_auc_score(yb, proba, average="macro", multi_class="ovr"))
                out["metrics"]["log_loss"] = _f(M.log_loss(y_true, proba, labels=labels))
                fpr, tpr, _ = M.roc_curve(yb.ravel(), proba.ravel())
                out["roc"] = _thin(fpr, tpr)
                out["roc_note"] = "Micro-averaged over all classes."
        except ValueError:
            pass
    out["how_to_read"] = {
        "accuracy": "Share of rows predicted correctly. Misleading when classes are imbalanced: predicting the majority class always scores high.",
        "balanced_accuracy": "Average recall per class. Fairer than accuracy on imbalanced data.",
        "precision": "Of the rows predicted positive, how many really were. High precision = few false alarms.",
        "recall": "Of the rows that really were positive, how many were caught. High recall = few misses.",
        "f1": "Harmonic mean of precision and recall. Use it when you care about both.",
        "roc_auc": "Probability that a random positive is ranked above a random negative. 0.5 is coin-flip, 1.0 is perfect.",
        "log_loss": "Penalizes confident wrong probabilities. Lower is better.",
        "mcc": "Correlation between predictions and truth, from -1 to 1. Robust to imbalance.",
        "confusion": "Rows are the true class, columns are the predicted class. The diagonal is correct; everything else is a specific kind of mistake.",
        "roc": "Each point is a probability threshold. Up and to the left is better. The diagonal is random guessing.",
        "pr": "Precision versus recall as the threshold moves. Better than ROC when positives are rare.",
    }
    return out


def regression(y_true, y_pred, n_features: int) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    r2 = M.r2_score(y_true, y_pred)
    adj = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    resid = y_true - y_pred
    idx = np.random.RandomState(0).choice(n, min(n, 400), replace=False)
    out = {
        "metrics": {
            "r2": _f(r2),
            "adjusted_r2": _f(adj),
            "mae": _f(M.mean_absolute_error(y_true, y_pred)),
            "rmse": _f(np.sqrt(M.mean_squared_error(y_true, y_pred))),
            "mse": _f(M.mean_squared_error(y_true, y_pred)),
            "mape": _f(M.mean_absolute_percentage_error(y_true, y_pred)) if np.all(np.abs(y_true) > 1e-9) else None,
            "max_error": _f(M.max_error(y_true, y_pred)),
        },
        "primary": "r2",
        "pred_vs_actual": [[_f(a), _f(p)] for a, p in zip(y_true[idx], y_pred[idx])],
        "residual_hist": _hist(resid),
        "how_to_read": {
            "r2": "Share of the target's variance the model explains. 1 is perfect, 0 is no better than predicting the mean, negative is worse than the mean.",
            "adjusted_r2": "R2 penalized for the number of features, so adding junk columns stops looking like progress.",
            "mae": "Average absolute miss, in the target's own units. Easy to explain to anyone.",
            "rmse": "Root of the mean squared miss. Punishes big misses more than MAE.",
            "mape": "Average percentage miss. Undefined when the target can be zero.",
            "pred_vs_actual": "Each dot is one test row. A perfect model puts every dot on the diagonal.",
            "residual_hist": "Distribution of actual minus predicted. Should be centred at zero and roughly symmetric. A skew or two humps means the model is systematically missing something.",
        },
    }
    return out


def clustering(X, labels) -> dict:
    labels = np.asarray(labels)
    uniq = set(labels.tolist())
    n_clusters = len(uniq - {-1})
    out = {"n_clusters": n_clusters, "noise": int((labels == -1).sum()), "metrics": {}}
    if 1 < n_clusters < len(X):
        mask = labels != -1
        try:
            out["metrics"]["silhouette"] = _f(M.silhouette_score(X[mask], labels[mask], sample_size=min(2000, int(mask.sum())), random_state=0))
            out["metrics"]["davies_bouldin"] = _f(M.davies_bouldin_score(X[mask], labels[mask]))
            out["metrics"]["calinski_harabasz"] = _f(M.calinski_harabasz_score(X[mask], labels[mask]))
        except ValueError:
            pass
    out["how_to_read"] = {
        "silhouette": "From -1 to 1. How much closer each row is to its own cluster than to the nearest other cluster. Above 0.5 is clear structure.",
        "davies_bouldin": "Average similarity of each cluster with its most similar one. Lower is better.",
        "calinski_harabasz": "Ratio of between-cluster to within-cluster spread. Higher is better. Only comparable across runs on the same data.",
    }
    return out


def _thin(x, y, n: int = 120) -> list[list[float]]:
    x = np.asarray(x); y = np.asarray(y)
    if len(x) > n:
        idx = np.linspace(0, len(x) - 1, n).astype(int)
        x, y = x[idx], y[idx]
    return [[_f(a), _f(b)] for a, b in zip(x, y)]


def _hist(values, bins: int = 25) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(values, bins=bins)
    return {"edges": [_f(e) for e in edges], "counts": counts.tolist()}


def feature_importance(pipe, feature_names: list[str], X_test, y_test, task: str, max_rows: int) -> dict | None:
    """Native importance when the model has it, otherwise permutation importance on a capped sample."""
    model = pipe.named_steps["model"]
    names = feature_names
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
            kind = "impurity"
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            kind = "coefficient"
        else:
            from sklearn.inspection import permutation_importance
            n = min(len(X_test), max_rows)
            idx = np.random.RandomState(0).choice(len(X_test), n, replace=False)
            Xs = X_test.iloc[idx]
            ys = np.asarray(y_test)[idx]
            scoring = "accuracy" if task == "classification" else "r2"
            r = permutation_importance(pipe, Xs, ys, n_repeats=3, random_state=0, scoring=scoring, n_jobs=1)
            # permutation on raw columns, names are the raw columns
            names = list(X_test.columns)
            imp = r.importances_mean
            kind = "permutation"
    except Exception:
        return None
    if len(imp) != len(names):
        names = [f"f{i}" for i in range(len(imp))]
    order = np.argsort(imp)[::-1][:20]
    note = {
        "impurity": "How much each feature reduced impurity across all tree splits. Biased toward high-cardinality features.",
        "coefficient": "Absolute size of the linear weight. Only comparable because the features were scaled.",
        "permutation": "How much the score drops when a column is shuffled. Model-agnostic and measured on test data, so it is the most honest of the three.",
    }[kind]
    return {"kind": kind, "note": note, "items": [{"feature": str(names[i]), "value": _f(imp[i])} for i in order]}
