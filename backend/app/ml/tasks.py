"""Detect whether a target column makes this classification or regression, with the reason."""
from __future__ import annotations

import pandas as pd


def detect_task(df: pd.DataFrame, target: str | None) -> dict:
    if not target:
        return {
            "task": "unsupervised",
            "reason": "No target column was chosen, so there is nothing to predict. "
            "The data can only be grouped (clustering) or compressed (dimensionality reduction).",
        }
    s = df[target]
    n = int(s.notna().sum())
    nunique = int(s.nunique(dropna=True))
    is_numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if not is_numeric:
        if nunique > 50:
            return {
                "task": "classification",
                "reason": f"'{target}' is text with {nunique} distinct values. That is many classes; "
                "results may be weak. Consider grouping rare labels.",
                "classes": nunique,
                "warning": "high_cardinality_target",
            }
        return {
            "task": "classification",
            "reason": f"'{target}' holds {nunique} distinct labels, so the model must pick a category. That is classification.",
            "classes": nunique,
        }
    if nunique <= 20 and nunique / max(n, 1) < 0.05:
        return {
            "task": "classification",
            "reason": f"'{target}' is numeric but has only {nunique} distinct values, so it behaves like category codes. "
            "Treated as classification. Switch to regression if these are real quantities.",
            "classes": nunique,
            "alt": "regression",
        }
    return {
        "task": "regression",
        "reason": f"'{target}' is a continuous number with {nunique} distinct values, so the model predicts a quantity. That is regression.",
        "alt": "classification" if nunique <= 20 else None,
    }
