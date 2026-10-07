"""Clustering, dimensionality reduction, anomaly detection. Every run returns 2D points for a plot."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor

from .. import config
from . import catalog, evaluate, preprocess


def _matrix(df: pd.DataFrame, params: dict, max_rows: int):
    cols = params.get("columns") or list(df.columns)
    X = df[[c for c in cols if c in df.columns]]
    X, _, sub = preprocess.subsample(X, None, max_rows, False)
    ct, info = preprocess.build(X, params.get("preprocess"))
    Z = ct.fit_transform(X)
    Z = np.asarray(Z, dtype=float)
    return X, Z, info, sub


def _points2d(Z, labels=None, max_points: int = 1500) -> dict:
    n = len(Z)
    idx = np.random.RandomState(0).choice(n, min(n, max_points), replace=False) if n > max_points else np.arange(n)
    if Z.shape[1] > 2:
        p = PCA(n_components=2, random_state=0)
        P2 = p.fit_transform(Z[idx])
        axis = f"PCA projection (explains {p.explained_variance_ratio_.sum()*100:.0f}% of variance)"
    elif Z.shape[1] == 2:
        P2 = Z[idx]
        axis = "The two features"
    else:
        P2 = np.column_stack([Z[idx, 0], np.zeros(len(idx))])
        axis = "Single feature"
    lab = labels[idx].tolist() if labels is not None else None
    return {"xy": [[round(float(a), 4), round(float(b), 4)] for a, b in P2], "labels": lab, "axis": axis}


def cluster(df, params: dict, progress, deadline) -> dict:
    algo = params.get("algorithm", "kmeans")
    X, Z, info, sub = _matrix(df, params, config.TRAIN_ROWS)
    narration = [{"step": "Prepare", "text": f"{len(X)} rows, {Z.shape[1]} features after preprocessing. Scaling matters a lot here: clustering is all about distances, and an unscaled column with big numbers would dominate every distance.", "code": "Z = preprocessor.fit_transform(X)"}]
    extra: dict = {}
    if algo == "kmeans":
        k = int(min(max(params.get("k", 3), 2), 20))
        progress("Sweeping k for elbow and silhouette", 0.2)
        sweep = []
        ks = list(range(2, min(11, len(X) - 1)))
        for kk in ks:
            km = KMeans(n_clusters=kk, n_init=4, random_state=config.RANDOM_STATE).fit(Z)
            sil = evaluate.M.silhouette_score(Z, km.labels_, sample_size=min(1500, len(Z)), random_state=0) if kk < len(Z) else None
            sweep.append({"k": kk, "inertia": round(float(km.inertia_), 3), "silhouette": round(float(sil), 4) if sil is not None else None})
        best_k = max(sweep, key=lambda s: s["silhouette"] or -1)["k"] if sweep else k
        progress(f"Fitting KMeans with k={k}", 0.6)
        model = KMeans(n_clusters=k, n_init=10, random_state=config.RANDOM_STATE).fit(Z)
        labels = model.labels_
        extra = {"sweep": sweep, "suggested_k": best_k, "centers": model.cluster_centers_[:, :10].round(3).tolist()}
        narration.append({"step": "Choose k", "text": f"The elbow plot shows inertia (within-cluster spread) falling as k grows; look for the bend. Silhouette peaks at k={best_k}, which is the data's own suggestion. You chose k={k}.", "code": f"KMeans(n_clusters={k}, n_init=10)"})
    elif algo == "dbscan":
        eps = float(params.get("eps", 0.5)); ms = int(params.get("min_samples", 5))
        model = DBSCAN(eps=eps, min_samples=ms).fit(Z)
        labels = model.labels_
        narration.append({"step": "Density", "text": f"eps={eps} is the neighbourhood radius and min_samples={ms} is how many neighbours make a core point. Rows that reach no core point are noise, label -1. If everything is noise, raise eps; if everything is one cluster, lower it.", "code": f"DBSCAN(eps={eps}, min_samples={ms})"})
    elif algo == "agglomerative":
        k = int(min(max(params.get("k", 3), 2), 20))
        link = params.get("linkage", "ward") if params.get("linkage") in {"ward", "complete", "average", "single"} else "ward"
        model = AgglomerativeClustering(n_clusters=k, linkage=link).fit(Z)
        labels = model.labels_
        narration.append({"step": "Merge", "text": f"Starting from {len(X)} singleton clusters, the closest pair merges repeatedly ({link} linkage decides what 'closest' means) until {k} remain.", "code": f"AgglomerativeClustering(n_clusters={k}, linkage='{link}')"})
    elif algo == "gmm":
        k = int(min(max(params.get("k", 3), 2), 20))
        model = GaussianMixture(n_components=k, random_state=config.RANDOM_STATE).fit(Z)
        labels = model.predict(Z)
        extra = {"bic": round(float(model.bic(Z)), 2)}
        narration.append({"step": "Mixture", "text": f"{k} Gaussian blobs are fitted by expectation-maximization. Unlike KMeans, each blob can be stretched, and each row has a soft membership. BIC = {extra['bic']:.1f}; lower BIC across different k means a better trade-off of fit and complexity.", "code": f"GaussianMixture(n_components={k})"})
    else:
        raise ValueError(f"Unknown algorithm '{algo}'.")
    ev = evaluate.clustering(Z, labels)
    sizes = pd.Series(labels).value_counts().sort_index()
    return {"algorithm": algo, "explain": catalog.UNSUPERVISED_TEXT.get(algo, ""), "rows_used": len(X), "subsampled": sub,
            "evaluation": ev, "sizes": [{"label": int(i), "count": int(c)} for i, c in sizes.items()],
            "points": _points2d(Z, np.asarray(labels)), "preprocessing": info, "narration": narration, **extra}


def reduce(df, params: dict, progress, deadline) -> dict:
    algo = params.get("algorithm", "pca")
    X, Z, info, sub = _matrix(df, params, config.TRAIN_ROWS if algo == "pca" else config.EMBED_ROWS)
    color = None
    if params.get("color_by") in df.columns:
        color = df.loc[X.index, params["color_by"]].astype(str).tolist()
    if algo == "pca":
        n = int(min(max(params.get("n_components", 2), 2), min(10, Z.shape[1])))
        p = PCA(n_components=n, random_state=0).fit(Z)
        T = p.transform(Z)
        ratio = p.explained_variance_ratio_.round(4).tolist()
        cum = np.cumsum(p.explained_variance_ratio_).round(4).tolist()
        loadings = []
        names = preprocess.feature_names(info and _ct_from(df, X, params)) or [f"f{i}" for i in range(Z.shape[1])]
        for ci in range(min(2, n)):
            comp = p.components_[ci]
            top = np.argsort(np.abs(comp))[::-1][:8]
            loadings.append([{"feature": str(names[i]) if i < len(names) else f"f{i}", "weight": round(float(comp[i]), 4)} for i in top])
        pts = {"xy": [[round(float(a), 4), round(float(b), 4)] for a, b in T[:1500, :2]], "labels": color[:1500] if color else None, "axis": "PC1 vs PC2"}
        narration = [{"step": "PCA", "text": f"{catalog.UNSUPERVISED_TEXT['pca']} The first two components explain {cum[1]*100:.0f}% of the variance. The loadings table shows which original features each component is made of.", "code": f"PCA(n_components={n}).fit_transform(Z)"}]
        return {"algorithm": "pca", "explained": ratio, "cumulative": cum, "loadings": loadings, "points": pts, "rows_used": len(X), "preprocessing": info, "narration": narration}
    if algo == "tsne":
        perp = float(min(max(params.get("perplexity", 30), 5), min(50, len(X) - 1)))
        progress("Running t-SNE (this is the slow one)", 0.3)
        T = TSNE(n_components=2, perplexity=perp, random_state=0, init="pca", max_iter=500).fit_transform(Z)
        pts = {"xy": [[round(float(a), 4), round(float(b), 4)] for a, b in T], "labels": color, "axis": "t-SNE 1 vs t-SNE 2"}
        narration = [{"step": "t-SNE", "text": f"{catalog.UNSUPERVISED_TEXT['tsne']} Perplexity {perp} roughly sets how many neighbours each point cares about. Capped at {config.EMBED_ROWS} rows.", "code": f"TSNE(n_components=2, perplexity={perp}, init='pca')"}]
        return {"algorithm": "tsne", "points": pts, "rows_used": len(X), "subsampled": sub, "preprocessing": info, "narration": narration}
    raise ValueError(f"Unknown algorithm '{algo}'.")


def _ct_from(df, X, params):
    ct, _ = preprocess.build(X, params.get("preprocess"))
    ct.fit(X)
    return ct


def anomaly(df, params: dict, progress, deadline) -> dict:
    algo = params.get("algorithm", "isoforest")
    contamination = float(min(max(params.get("contamination", 0.05), 0.005), 0.3))
    X, Z, info, sub = _matrix(df, params, config.TRAIN_ROWS)
    if algo == "isoforest":
        model = IsolationForest(contamination=contamination, random_state=config.RANDOM_STATE, n_jobs=1).fit(Z)
        score = -model.score_samples(Z)
        flag = model.predict(Z) == -1
    elif algo == "lof":
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
        flag = model.fit_predict(Z) == -1
        score = -model.negative_outlier_factor_
    else:
        raise ValueError(f"Unknown algorithm '{algo}'.")
    order = np.argsort(score)[::-1][:25]
    top = X.iloc[order].copy()
    top.insert(0, "_score", np.round(score[order], 4))
    rows = top.astype(object).where(top.notna(), None).values.tolist()
    narration = [{"step": "Detect", "text": f"{catalog.UNSUPERVISED_TEXT[algo]} contamination={contamination} tells the model to flag roughly {contamination*100:.1f}% of rows.", "code": f"{'IsolationForest' if algo == 'isoforest' else 'LocalOutlierFactor'}(contamination={contamination})"}]
    return {"algorithm": algo, "flagged": int(flag.sum()), "rows_used": len(X), "subsampled": sub, "columns": list(top.columns), "top": rows,
            "points": _points2d(Z, flag.astype(int)), "preprocessing": info, "narration": narration}
