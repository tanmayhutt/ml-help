"""Built-in teaching datasets, loaded from sklearn (no download, no network)."""
from __future__ import annotations

import pandas as pd
from sklearn import datasets as skd

SAMPLES = {
    "iris": {"name": "Iris flowers", "task": "classification", "target": "species", "about": "150 flowers, 4 measurements each, 3 species. The classic beginner file. Easy: most models score above 90%."},
    "wine": {"name": "Wine cultivars", "task": "classification", "target": "cultivar", "about": "178 wines, 13 measurements, 3 producers. Shows why scaling matters: columns range from 0.1 to 1500."},
    "breast_cancer": {"name": "Breast cancer diagnosis", "task": "classification", "target": "diagnosis", "about": "569 tumours, 30 measurements, harmless or harmful. A realistic yes/no problem."},
    "digits": {"name": "Handwritten digits", "task": "classification", "target": "digit", "about": "1797 tiny images of handwritten digits, 64 pixel columns, 10 digits. Bigger and harder."},
    "diabetes": {"name": "Diabetes progression", "task": "regression", "target": "progression", "about": "442 patients, 10 measurements, a disease score to predict. Hard: even the best model only explains about half of it."},
    "california": {"name": "California housing (sample)", "task": "regression", "target": "median_house_value", "about": "5000 neighbourhoods with a house price to predict. Tree models beat straight-line models clearly here."},
}


def load(key: str) -> pd.DataFrame:
    if key == "iris":
        d = skd.load_iris(as_frame=True); df = d.frame; df["species"] = d.target_names[d.target]; df = df.drop(columns=["target"])
        df.columns = [c.replace(" (cm)", "_cm").replace(" ", "_") for c in df.columns]
    elif key == "wine":
        d = skd.load_wine(as_frame=True); df = d.frame; df["cultivar"] = d.target_names[d.target]; df = df.drop(columns=["target"])
    elif key == "breast_cancer":
        d = skd.load_breast_cancer(as_frame=True); df = d.frame; df["diagnosis"] = d.target_names[d.target]; df = df.drop(columns=["target"])
        df.columns = [c.replace(" ", "_") for c in df.columns]
    elif key == "digits":
        d = skd.load_digits(as_frame=True); df = d.frame.rename(columns={"target": "digit"}); df["digit"] = df["digit"].astype(str)
    elif key == "diabetes":
        d = skd.load_diabetes(as_frame=True); df = d.frame.rename(columns={"target": "progression"})
    elif key == "california":
        try:
            d = skd.fetch_california_housing(as_frame=True, download_if_missing=False)
        except Exception:
            raise ValueError("California housing is not cached on this server. Pick another sample.")
        df = d.frame.rename(columns={"MedHouseVal": "median_house_value"}).sample(5000, random_state=0).reset_index(drop=True)
    else:
        raise ValueError("Unknown sample.")
    return df
