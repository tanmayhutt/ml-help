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
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                out_vals = v[(v < lo) | (v > hi)]
                info["box"] = {"min": _f(max(v.min(), lo)), "q1": _f(q1), "median": _f(v.median()), "q3": _f(q3), "max": _f(min(v.max(), hi)),
                               "outliers": [_f(x) for x in out_vals.sample(min(len(out_vals), 40), random_state=0)], "n_outliers": int(len(out_vals))}
                if len(v) > 10 and v.min() >= 0 and abs(float(v.skew())) >= 1.0:
                    info["skewed"] = True
            if nunique <= 10 and n > 50:
                info["looks_categorical"] = True
            name_l = str(c).lower()
            if n > 20 and nunique >= 0.98 * n and (pd.api.types.is_integer_dtype(s) or name_l in ("id", "index") or name_l.endswith("_id") or name_l.endswith("id")):
                info["looks_id"] = True
        elif pd.api.types.is_datetime64_any_dtype(s):
            info["type"] = "datetime"
        else:
            info["type"] = "text"
            info["top"] = _top(s)
            if _looks_like_dates(s):
                info["type"] = "datetime"
                info.pop("top", None)
            elif n > 20 and nunique >= 0.98 * n:
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
    id_like = {c["name"] for c in cols if c.get("looks_id")}
    numeric_cols = [c for c in numeric_cols if c not in id_like]
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
    duplicates = int(df.duplicated().sum())
    if duplicates:
        notes.append({"kind": "info", "text": f"{duplicates} rows are exact duplicates of another row. The cleaning plan removes them."})
    pairs = []
    if corr:
        cm = pd.DataFrame(corr["matrix"], index=numeric_cols, columns=numeric_cols)
        cand = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                cand.append((abs(cm.iat[i, j]), numeric_cols[i], numeric_cols[j], cm.iat[i, j]))
        cand.sort(reverse=True)
        samp = df.sample(min(n, 300), random_state=0)
        for _, a, b, r in cand[:3]:
            pts = samp[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            pairs.append({"x": a, "y": b, "r": _f(r), "points": [[_f(u), _f(w)] for u, w in pts.values.tolist()]})
    head = df.head(15).astype(object).where(df.head(15).notna(), None)
    suggested = suggest_targets(df, cols)
    plan = cleaning_plan(df, cols, duplicates, corr, numeric_cols, suggested[0]["column"] if suggested else None)
    return {
        "rows": n, "cols": len(df.columns), "columns": cols, "notes": notes, "correlation": corr, "suggested_targets": suggested,
        "duplicates": duplicates, "pairs": pairs, "plots": plot_code(df, cols, numeric_cols, suggested), "cleaning": plan,
        "head": {"columns": list(df.columns), "rows": [[_cell(v) for v in r] for r in head.values.tolist()]},
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
    }


def _looks_like_dates(s: pd.Series) -> bool:
    sample = s.dropna().astype(str).head(200)
    if len(sample) < 5:
        return False
    if not sample.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}", regex=True).mean() > 0.9:
        return False
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return False
    return bool(parsed.notna().mean() > 0.9)


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


def plot_code(df: pd.DataFrame, cols: list[dict], numeric_cols: list[str], suggested: list[dict]) -> list[dict]:
    """The seaborn / matplotlib code that draws each chart shown on the page."""
    cat_cols = [c["name"] for c in cols if c["type"] in ("text", "boolean") and c["unique"] <= 20]
    target = suggested[0]["column"] if suggested else "target"
    plots = [
        {"title": "Load and look", "what": "Shape, column types, blanks, and basic statistics.", "code": "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ndf = pd.read_csv('your_dataset.csv')\nprint(df.shape)\nprint(df.dtypes)\nprint(df.isna().sum())\nprint(df.describe(include='all').T)\nprint('duplicates:', df.duplicated().sum())"},
        {"title": "Histogram of every number column", "what": "The shape of each column: bell, skewed, two humps, or spikes.", "code": "df[" + repr(numeric_cols) + "].hist(bins=20, figsize=(14, 10))\nplt.tight_layout(); plt.show()"},
        {"title": "Box plot of every number column", "what": "Median, quartiles, and the dots beyond the whiskers are outliers.", "code": "fig, axes = plt.subplots(1, len(" + repr(numeric_cols) + "), figsize=(3 * len(" + repr(numeric_cols) + "), 4))\nfor ax, col in zip(np.atleast_1d(axes), " + repr(numeric_cols) + "):\n    sns.boxplot(y=df[col], ax=ax)\nplt.tight_layout(); plt.show()"},
        {"title": "Missing values", "what": "How many blanks each column has.", "code": "missing = df.isna().mean().sort_values(ascending=False) * 100\nmissing[missing > 0].plot.barh(figsize=(8, 5), title='% missing per column')\nplt.show()\n\n# or as a heatmap of where the blanks are\nsns.heatmap(df.isna(), cbar=False)\nplt.show()"},
    ]
    if cat_cols:
        plots.append({"title": "Count plot of every category column", "what": "How many rows fall in each category. Reveals rare categories and imbalance.", "code": "for col in " + repr(cat_cols) + ":\n    sns.countplot(y=df[col], order=df[col].value_counts().index)\n    plt.title(col); plt.show()"})
    if len(numeric_cols) >= 2:
        plots.append({"title": "Correlation heatmap", "what": "Which number columns move together. Values near 1 or -1 mean two columns carry the same information.", "code": "plt.figure(figsize=(10, 8))\nsns.heatmap(df[" + repr(numeric_cols) + "].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)\nplt.show()"})
        few = numeric_cols[:6]
        plots.append({"title": "Pair plot", "what": "Every number column against every other, coloured by the answer column. Shows which columns separate the groups.", "code": "sns.pairplot(df[" + repr(few + ([target] if target in df.columns and target not in few else [])) + "]" + (f", hue='{target}'" if target in df.columns and target not in numeric_cols else "") + ", corner=True)\nplt.show()"})
    if target in df.columns:
        if target in numeric_cols and df[target].nunique() > 20:
            plots.append({"title": "Answer column distribution", "what": "The spread of the value you want to predict. A long tail suggests predicting log(value) instead.", "code": f"sns.histplot(df['{target}'], kde=True)\nplt.show()\nprint('skew:', df['{target}'].skew())"})
        else:
            plots.append({"title": "Class balance of the answer column", "what": "How many rows per class. A very unequal split means accuracy alone is misleading; look at balanced accuracy or F1.", "code": f"df['{target}'].value_counts().plot.bar(title='rows per class')\nplt.show()\nprint(df['{target}'].value_counts(normalize=True).round(3))"})
            if numeric_cols:
                plots.append({"title": "Each number column by class", "what": "Box plots split by class. Columns whose boxes do not overlap are the strong predictors.", "code": f"for col in {numeric_cols[:8]!r}:\n    sns.boxplot(x=df['{target}'], y=df[col])\n    plt.show()"})
    return plots


def cleaning_plan(df: pd.DataFrame, cols: list[dict], duplicates: int, corr: dict | None, numeric_cols: list[str], target: str | None = None) -> dict:
    """Recommended cleaning steps with reasons and pandas code, plus the preprocessing options that apply them."""
    steps = []
    drop = []
    options = {"num_impute": "median", "cat_impute": "most_frequent", "scaler": "standard", "encoder": "onehot", "drop_columns": [], "drop_duplicates": False, "outliers": "none", "log_skewed": False}
    for c in cols:
        if c["name"] == target:
            continue
        if c.get("looks_id"):
            drop.append(c["name"]); steps.append({"step": f"Drop '{c['name']}'", "why": "Every value is different, so it is an ID or an index. It cannot help predict anything and a tree would just memorise it.", "code": f"df = df.drop(columns=['{c['name']}'])"})
        elif c["type"] == "datetime":
            drop.append(c["name"]); steps.append({"step": f"Drop or expand '{c['name']}'", "why": "Dates cannot be used as-is. Either drop the column, or turn it into useful numbers like year, month, weekday, or days since a reference date.", "code": f"df['{c['name']}'] = pd.to_datetime(df['{c['name']}'])\ndf['{c['name']}_year'] = df['{c['name']}'].dt.year\ndf['{c['name']}_month'] = df['{c['name']}'].dt.month\ndf['{c['name']}_weekday'] = df['{c['name']}'].dt.weekday\ndf = df.drop(columns=['{c['name']}'])"})
        elif c["missing_pct"] > 40:
            drop.append(c["name"]); steps.append({"step": f"Drop '{c['name']}'", "why": f"{c['missing_pct']}% of it is blank. Filling that many blanks would invent most of the column.", "code": f"df = df.drop(columns=['{c['name']}'])"})
        elif c["unique"] <= 1:
            drop.append(c["name"]); steps.append({"step": f"Drop '{c['name']}'", "why": "Every row has the same value, so it carries no information.", "code": f"df = df.drop(columns=['{c['name']}'])"})
        elif c["type"] == "text" and c["unique"] > 50:
            drop.append(c["name"]); steps.append({"step": f"Drop '{c['name']}' (or group its rare values)", "why": f"{c['unique']} different text values is too many to one-hot encode. It is probably free text, a name, or an ID. If the frequent values matter, keep the top 10 and label the rest 'other'.", "code": f"top = df['{c['name']}'].value_counts().index[:10]\ndf['{c['name']}'] = df['{c['name']}'].where(df['{c['name']}'].isin(top), 'other')\n# or simply: df = df.drop(columns=['{c['name']}'])"})
    if duplicates:
        options["drop_duplicates"] = True
        steps.append({"step": f"Remove {duplicates} duplicate rows", "why": "Exact copies add nothing and can leak the same row into both training and test sets, inflating the score.", "code": "df = df.drop_duplicates()"})
    missing_num = [c["name"] for c in cols if c["type"] == "numeric" and 0 < c["missing_pct"] <= 40 and c["name"] != target]
    missing_cat = [c["name"] for c in cols if c["type"] in ("text", "boolean") and 0 < c["missing_pct"] <= 40 and c["name"] != target]
    if missing_num:
        steps.append({"step": "Fill blank numbers with the median", "why": "Models cannot handle blanks. The median is a safe filler because a few extreme values do not pull it around. Learn the fill value from the training rows only.", "code": f"for col in {missing_num!r}:\n    df[col] = df[col].fillna(df[col].median())"})
    if missing_cat:
        steps.append({"step": "Fill blank categories with the most common value", "why": "Or use the word 'missing' as its own category if being blank might itself mean something.", "code": f"for col in {missing_cat!r}:\n    df[col] = df[col].fillna(df[col].mode()[0])"})
    outlier_cols = [c["name"] for c in cols if c.get("box") and c["box"]["n_outliers"] > 0 and c["name"] not in drop and c["name"] != target]
    if outlier_cols:
        options["outliers"] = "cap"
        steps.append({"step": "Cap outliers in " + ", ".join(outlier_cols[:6]) + (" and more" if len(outlier_cols) > 6 else ""), "why": "Values far beyond the usual range drag linear models and distance-based models around. Capping pulls them back to the edge of the normal range without deleting rows. Tree models do not need this.", "code": f"for col in {outlier_cols!r}:\n    q1, q3 = df[col].quantile([0.25, 0.75])\n    iqr = q3 - q1\n    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)"})
    skewed = [c["name"] for c in cols if c.get("skewed") and c["name"] not in drop and c["name"] != target]
    if skewed:
        options["log_skewed"] = True
        steps.append({"step": "Log-transform skewed columns: " + ", ".join(skewed[:6]), "why": "These have a long tail of large values. log(1 + x) squashes the tail so the spread is more even, which helps linear models and neural networks.", "code": f"for col in {skewed!r}:\n    df[col] = np.log1p(df[col])"})
    cats = [c for c in cols if c["type"] in ("text", "boolean") and c["name"] not in drop and c["name"] != target]
    small = [c["name"] for c in cats if c["unique"] <= 10]
    medium = [c["name"] for c in cats if 10 < c["unique"] <= 50]
    if small:
        steps.append({"step": "One-hot encode " + ", ".join(small[:6]) + (" and more" if len(small) > 6 else ""), "why": "Each category becomes its own 0/1 column. The model never assumes an order between categories like red < green < blue.", "code": f"df = pd.get_dummies(df, columns={small!r}, drop_first=False)"})
    if medium:
        steps.append({"step": "Encode " + ", ".join(medium) + " with care", "why": "Between 10 and 50 categories: one-hot still works but creates many columns. For tree models an ordinal code is fine; for linear models prefer one-hot or group rare categories first.", "code": f"from sklearn.preprocessing import OrdinalEncoder\nenc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\ndf[{medium!r}] = enc.fit_transform(df[{medium!r}])"})
    ordered = [c["name"] for c in cats if any(w in c["name"].lower() for w in ("size", "level", "grade", "rating", "rank", "tier", "class"))]
    if ordered:
        steps.append({"step": "Check whether " + ", ".join(ordered) + " has a natural order", "why": "Words like small/medium/large or low/high have an order. If so, map them to numbers by hand instead of one-hot encoding, so the model can use the order.", "code": f"order = {{'low': 0, 'medium': 1, 'high': 2}}  # edit to match your values\ndf['{ordered[0]}'] = df['{ordered[0]}'].map(order)"})
    if corr:
        strong = []
        cm = corr["matrix"]
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if abs(cm[i][j]) > 0.9:
                    strong.append((numeric_cols[i], numeric_cols[j], cm[i][j]))
        if strong:
            steps.append({"step": "Consider dropping one of each near-duplicate pair", "why": "; ".join(f"{a} and {b} ({r:.2f})" for a, b, r in strong[:4]) + ". Two columns that say the same thing make linear models unstable and double-count that signal. Trees do not mind.", "code": "df = df.drop(columns=[" + ", ".join(repr(b) for _, b, _ in strong[:4]) + "])"})
    steps.append({"step": "Scale the number columns", "why": "Put every number column on the same scale (mean 0, spread 1) so that no column dominates just because its numbers are bigger. Essential for KNN, SVM, logistic regression and neural networks. Fit the scaler on training rows only.", "code": "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train = scaler.fit_transform(X_train)\nX_test = scaler.transform(X_test)"})
    steps.append({"step": "Do all of it inside a pipeline", "why": "A scikit-learn Pipeline runs every step above on the training rows, remembers the fill values, caps and scales, and reapplies them to new rows. That is how the tool does it, so nothing leaks from the test rows into training.", "code": "from sklearn.pipeline import Pipeline\nfrom sklearn.compose import ColumnTransformer\npipe = Pipeline([('prep', preprocessor), ('model', model)])\npipe.fit(X_train, y_train)"})
    if target:
        steps.insert(0, {"step": f"Keep '{target}' as the answer column", "why": "It is the thing to predict, so it is never cleaned, encoded or scaled with the other columns. Rows where it is blank are removed instead.", "code": f"df = df[df['{target}'].notna()]\ny = df['{target}']\nX = df.drop(columns=['{target}'])"})
    options["drop_columns"] = drop
    script = "import pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('your_dataset.csv')\n\n" + "\n\n".join(f"# {st['step']}\n# {st['why']}\n{st['code']}" for st in steps if not st["step"].startswith("Do all of it") and not st["step"].startswith("Scale"))
    return {"steps": steps, "options": options, "script": script}
