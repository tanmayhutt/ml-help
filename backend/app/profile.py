"""Dataset profile: types, missing values, stats, histograms, correlations, with plain-English notes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _f(x):
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


def profile(df: pd.DataFrame) -> dict:
    n = len(df)
    cols = []
    notes = []
    numeric_cols = []
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        info: dict = {"name": c, "missing": missing, "missing_pct": round(100 * missing / max(n, 1), 1), "unique": nunique}
        if pd.api.types.is_bool_dtype(s):
            info["type"] = "boolean"
            info["top"] = _top(s)
        elif pd.api.types.is_numeric_dtype(s):
            info["type"] = "numeric"
            numeric_cols.append(c)
            v = pd.to_numeric(s, errors="coerce").dropna().astype(float)
            if len(v):
                info.update({"mean": _f(v.mean()), "std": _f(v.std()), "min": _f(v.min()), "q25": _f(v.quantile(0.25)), "median": _f(v.median()), "q75": _f(v.quantile(0.75)), "max": _f(v.max()), "skew": _f(v.skew()) if len(v) > 2 else None})
                counts, edges = np.histogram(v, bins=20)
                info["hist"] = {"edges": [_f(e) for e in edges], "counts": counts.tolist()}
                iqr = v.quantile(0.75) - v.quantile(0.25)
                if iqr > 0:
                    out = int(((v < v.quantile(0.25) - 3 * iqr) | (v > v.quantile(0.75) + 3 * iqr)).sum())
                    info["outliers"] = out
            if nunique <= 10 and n > 50:
                info["looks_categorical"] = True
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["type"] = "datetime"
        else:
            info["type"] = "text"
            info["top"] = _top(s)
            if nunique == n and n > 20:
                info["looks_id"] = True
        cols.append(info)
    # Notes
    high_missing = [c for c in cols if c["missing_pct"] > 30]
    if high_missing:
        notes.append({"kind": "warn", "text": f"{', '.join(c['name'] for c in high_missing[:5])}: more than 30% missing. Imputation will invent a lot of values; consider dropping these columns."})
    ids = [c["name"] for c in cols if c.get("looks_id")]
    if ids:
        notes.append({"kind": "warn", "text": f"{', '.join(ids[:5])}: every value is unique, so this looks like an ID. It carries no pattern and will be dropped automatically."})
    skewed = [c["name"] for c in cols if c.get("skew") is not None and abs(c["skew"]) > 2]
    if skewed:
        notes.append({"kind": "info", "text": f"{', '.join(skewed[:5])}: heavily skewed. A log transform or the robust scaler often helps linear models here. Trees do not care."})
    cat_like = [c["name"] for c in cols if c.get("looks_categorical")]
    if cat_like:
        notes.append({"kind": "info", "text": f"{', '.join(cat_like[:5])}: numeric with very few distinct values. These may really be categories."})
    if n < 100:
        notes.append({"kind": "warn", "text": f"Only {n} rows. Expect scores to swing a lot between folds; trust cross-validation, not a single split."})
    corr = None
    if 2 <= len(numeric_cols) <= 30:
        sub = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if n > 5000:
            sub = sub.sample(5000, random_state=0)
        cm = sub.corr().fillna(0).round(3)
        corr = {"columns": numeric_cols, "matrix": cm.values.tolist()}
        strong = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if abs(cm.iat[i, j]) > 0.9:
                    strong.append(f"{numeric_cols[i]} and {numeric_cols[j]} ({cm.iat[i, j]:.2f})")
        if strong:
            notes.append({"kind": "info", "text": "Nearly duplicate columns: " + "; ".join(strong[:4]) + ". Linear models get unstable with these; Ridge or dropping one helps."})
    head = df.head(15).astype(object).where(df.head(15).notna(), None)
    return {
        "rows": n, "cols": len(df.columns), "columns": cols, "notes": notes, "correlation": corr, "suggested_targets": suggest_targets(df, cols),
        "head": {"columns": list(df.columns), "rows": [[_cell(v) for v in r] for r in head.values.tolist()]},
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def _top(s: pd.Series, k: int = 6) -> list[dict]:
    vc = s.astype(str).value_counts().head(k)
    return [{"value": str(i)[:40], "count": int(c)} for i, c in vc.items()]


def _cell(v):
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return _f(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)[:80]


TARGET_WORDS = ("target", "label", "class", "outcome", "result", "y", "price", "score", "species", "survived", "churn", "diagnosis", "category", "type", "status", "default", "fraud", "rating", "salary", "value", "sales", "amount", "quality", "grade", "risk", "winner", "approved", "clicked", "purchased", "income", "disease", "progression")


def suggest_targets(df: pd.DataFrame, cols: list[dict]) -> list[dict]:
    """Rank columns by how likely they are to be the thing a person wants to predict. Heuristic, explained."""
    n = len(df)
    out = []
    for i, c in enumerate(cols):
        score = 0.0
        why = []
        name = c["name"].lower()
        if c.get("looks_id") or c["type"] == "datetime":
            continue
        if c["missing_pct"] > 40:
            continue
        if c["unique"] <= 1:
            continue
        if i == len(cols) - 1:
            score += 2; why.append("it is the last column")
        if any(w == name or name.endswith("_" + w) or name.startswith(w + "_") or w in name.split() for w in TARGET_WORDS):
            score += 3; why.append("its name sounds like an answer")
        if c["type"] in ("text", "boolean") and 2 <= c["unique"] <= 20:
            score += 2; why.append(f"it has {c['unique']} categories")
        if c["type"] == "numeric" and c.get("looks_categorical"):
            score += 1; why.append("it looks like a code with few values")
        if c["type"] == "numeric" and not c.get("looks_categorical"):
            score += 0.5
        if c["type"] == "text" and c["unique"] > 50:
            score -= 2
        kind = "category" if (c["type"] in ("text", "boolean") or c.get("looks_categorical")) else "number"
        out.append({"column": c["name"], "score": round(score, 2), "kind": kind, "why": ", ".join(why) or "a plain column"})
    out.sort(key=lambda d: -d["score"])
    return out[:5]
