"""Build the preprocessing pipeline and describe it in plain language."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap values outside [Q1 - k*IQR, Q3 + k*IQR]. Limits are learned on training rows only."""

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X, 25, axis=0)
        q3 = np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else None

NUM_IMPUTE = {"mean", "median", "most_frequent", "constant"}
CAT_IMPUTE = {"most_frequent", "constant"}
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler, "none": None}
ENCODERS = {"onehot", "ordinal"}

DEFAULTS = {
    "num_impute": "median",
    "cat_impute": "most_frequent",
    "scaler": "standard",
    "encoder": "onehot",
    "drop_high_cardinality": 50,
    "drop_columns": [],
    "drop_duplicates": False,
    "outliers": "none",      # none | cap
    "log_skewed": False,     # log1p on skewed, non-negative numeric columns
    "skew_threshold": 1.0,
}


def normalize(options: dict | None) -> dict:
    o = dict(DEFAULTS)
    o.update({k: v for k, v in (options or {}).items() if k in DEFAULTS})
    if o["num_impute"] not in NUM_IMPUTE:
        o["num_impute"] = "median"
    if o["cat_impute"] not in CAT_IMPUTE:
        o["cat_impute"] = "most_frequent"
    if o["scaler"] not in SCALERS:
        o["scaler"] = "standard"
    if o["encoder"] not in ENCODERS:
        o["encoder"] = "onehot"
    o["drop_high_cardinality"] = int(o["drop_high_cardinality"] or 0)
    o["drop_columns"] = [str(c) for c in (o["drop_columns"] or [])]
    o["drop_duplicates"] = bool(o["drop_duplicates"])
    o["outliers"] = o["outliers"] if o["outliers"] in ("none", "cap") else "none"
    o["log_skewed"] = bool(o["log_skewed"])
    o["skew_threshold"] = float(o["skew_threshold"] or 1.0)
    return o


def split_columns(X: pd.DataFrame, options: dict) -> dict:
    """Decide which columns are numeric, categorical, or dropped, and why."""
    numeric, categorical, dropped = [], [], []
    for c in X.columns:
        s = X[c]
        if c in options["drop_columns"]:
            dropped.append({"column": c, "reason": "dropped by you"})
            continue
        if s.isna().all():
            dropped.append({"column": c, "reason": "every value is missing"})
            continue
        if pd.api.types.is_bool_dtype(s):
            categorical.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1:
                dropped.append({"column": c, "reason": "constant column carries no information"})
            else:
                numeric.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            dropped.append({"column": c, "reason": "datetime columns need feature engineering first (year, month, weekday)"})
            continue
        nunique = s.nunique(dropna=True)
        limit = options["drop_high_cardinality"]
        if limit and nunique > limit:
            dropped.append({"column": c, "reason": f"{nunique} distinct text values is too many to one-hot encode (limit {limit}); looks like an ID or free text"})
        elif nunique <= 1:
            dropped.append({"column": c, "reason": "constant column carries no information"})
        else:
            categorical.append(c)
    log_cols: list[str] = []
    if options.get("log_skewed"):
        for c in numeric:
            v = pd.to_numeric(X[c], errors="coerce").dropna().astype(float)
            if len(v) > 10 and v.min() >= 0 and abs(float(v.skew())) >= options.get("skew_threshold", 1.0):
                log_cols.append(c)
    numeric = [c for c in numeric if c not in log_cols]
    return {"numeric": numeric, "numeric_log": log_cols, "categorical": categorical, "dropped": dropped}


def _num_steps(options: dict, log: bool) -> list:
    steps = [("impute", SimpleImputer(strategy=options["num_impute"], fill_value=0))]
    if options.get("outliers") == "cap":
        steps.append(("cap", Winsorizer()))
    if log:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    scaler_cls = SCALERS[options["scaler"]]
    if scaler_cls is not None:
        steps.append(("scale", scaler_cls()))
    return steps


def build(X: pd.DataFrame, options: dict | None) -> tuple[ColumnTransformer, dict]:
    options = normalize(options)
    cols = split_columns(X, options)
    if options["encoder"] == "onehot":
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=1)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [
        ("impute", SimpleImputer(strategy=options["cat_impute"], fill_value="missing")),
        ("encode", enc),
    ]
    transformers = []
    if cols["numeric"]:
        transformers.append(("num", Pipeline(_num_steps(options, False)), cols["numeric"]))
    if cols.get("numeric_log"):
        transformers.append(("num_log", Pipeline(_num_steps(options, True)), cols["numeric_log"]))
    if cols["categorical"]:
        transformers.append(("cat", Pipeline(cat_steps), cols["categorical"]))
    if not transformers:
        raise ValueError("No usable feature columns remain after dropping constants, IDs and empty columns.")
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    info = {"options": options, "columns": cols, "steps": describe(options, cols)}
    return ct, info


def describe(options: dict, cols: dict) -> list[dict]:
    steps = []
    all_num = cols["numeric"] + cols.get("numeric_log", [])
    if options.get("drop_duplicates"):
        steps.append({"title": "Remove duplicate rows", "why": "Exact copies of a row teach the model nothing new and can leak the same row into both the training and the test side.", "code": "df = df.drop_duplicates()", "columns": []})
    if all_num:
        steps.append({
            "title": f"Fill missing numbers with the {options['num_impute'].replace('_', ' ')}",
            "why": "Most models cannot handle blanks. The median is robust to outliers; the mean is fine for symmetric data.",
            "code": f"SimpleImputer(strategy='{options['num_impute']}')",
            "columns": all_num,
        })
    if all_num and options.get("outliers") == "cap":
        steps.append({"title": "Cap extreme values", "why": "Values far outside the usual range (beyond 1.5 times the interquartile range) are pulled back to the edge of that range. The limits are learned on training rows only. Trees do not care about outliers, but linear models and distance models do.", "code": "Winsorizer(k=1.5)  # clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]", "columns": all_num})
    if cols.get("numeric_log"):
        steps.append({"title": "Log-transform skewed columns", "why": "These columns have a long tail (a few very large values). Taking log(1 + x) squashes the tail so the model sees a more even spread.", "code": "np.log1p(df[cols])", "columns": cols["numeric_log"]})
        if options["scaler"] != "none":
            why = {
                "standard": "Rescales each number column to mean 0 and spread 1. Distance-based models (KNN, SVM) and gradient models (logistic regression, neural nets) need features on the same scale. Trees do not care.",
                "minmax": "Squeezes every number column into the 0 to 1 range. Useful when you need bounded inputs, but sensitive to outliers.",
                "robust": "Scales using the median and interquartile range, so a few extreme values do not dominate.",
            }[options["scaler"]]
            steps.append({"title": f"Scale numbers ({options['scaler']})", "why": why, "code": f"{SCALERS[options['scaler']].__name__}()", "columns": all_num})
    if cols["categorical"]:
        steps.append({
            "title": f"Fill missing categories with the {options['cat_impute'].replace('_', ' ')} value",
            "why": "A blank category becomes the most common label (or the literal word 'missing').",
            "code": f"SimpleImputer(strategy='{options['cat_impute']}')",
            "columns": cols["categorical"],
        })
        if options["encoder"] == "onehot":
            steps.append({"title": "One-hot encode text columns", "why": "Each category becomes its own 0/1 column, so the model never assumes an order between labels like red < green < blue.", "code": "OneHotEncoder(handle_unknown='ignore')", "columns": cols["categorical"]})
        else:
            steps.append({"title": "Ordinal encode text columns", "why": "Each category becomes an integer. Compact, but it invents an order. Fine for tree models, risky for linear ones.", "code": "OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)", "columns": cols["categorical"]})
    if cols["dropped"]:
        steps.append({"title": "Drop unusable columns", "why": "; ".join(f"{d['column']}: {d['reason']}" for d in cols["dropped"]), "code": "ColumnTransformer(remainder='drop')", "columns": [d["column"] for d in cols["dropped"]]})
    steps.append({
        "title": "Fit the preprocessing only on training rows",
        "why": "If the scaler or imputer saw the test rows, information would leak from test to train and the score would be too optimistic. This is the leakage rule.",
        "code": "Pipeline([('prep', preprocessor), ('model', model)]).fit(X_train, y_train)",
        "columns": [],
    })
    return steps


def feature_names(ct: ColumnTransformer) -> list[str]:
    try:
        return [str(n) for n in ct.get_feature_names_out()]
    except Exception:
        return []


def subsample(X: pd.DataFrame, y: pd.Series | None, max_rows: int, stratify: bool, random_state: int = 42):
    """Return at most max_rows rows. Stratified for classification so rare classes survive."""
    if len(X) <= max_rows:
        return X, y, False
    if y is not None and stratify:
        from sklearn.model_selection import train_test_split
        try:
            Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, stratify=y, random_state=random_state)
            return Xs, ys, True
        except ValueError:
            pass
    idx = np.random.RandomState(random_state).choice(len(X), max_rows, replace=False)
    Xs = X.iloc[idx]
    ys = y.iloc[idx] if y is not None else None
    return Xs, ys, True
