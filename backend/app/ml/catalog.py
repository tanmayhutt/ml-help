"""The model shelf. Every entry carries a plain-language explanation, a cost rating,
and a small random-search space. Ensembles are built from these base models."""
from __future__ import annotations

from typing import Any

from scipy.stats import loguniform, randint, uniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .. import config

RS = config.RANDOM_STATE


def _nj() -> int:
    return config.N_JOBS


# family: linear | distance | probabilistic | tree | kernel | neural | bagging | boosting
# cost: 1 (instant) .. 5 (slow). Models with slow=True are skipped in the race above SLOW_MODEL_ROWS.
CLASSIFIERS: dict[str, dict[str, Any]] = {
    "logreg": {
        "name": "Logistic Regression", "family": "linear", "cost": 1,
        "make": lambda: LogisticRegression(max_iter=1000, random_state=RS),
        "space": {"model__C": loguniform(1e-3, 1e2)},
        "explain": "Draws a straight boundary between classes and outputs a probability. Fast, stable, easy to interpret. Needs scaled features. Struggles when the boundary is curved.",
        "params": {"C": "Inverse regularization. Small C = simpler, more regularized model."},
    },
    "knn": {
        "name": "K-Nearest Neighbors", "family": "distance", "cost": 2, "slow": True,
        "make": lambda: KNeighborsClassifier(n_neighbors=5),
        "space": {"model__n_neighbors": randint(3, 31), "model__weights": ["uniform", "distance"]},
        "explain": "Looks at the k closest training rows and takes a vote. No training step at all; all the work happens at prediction time. Needs scaling. Slow on big data.",
        "params": {"n_neighbors": "How many neighbours vote. Small k = jagged boundary, large k = smooth."},
    },
    "nb": {
        "name": "Gaussian Naive Bayes", "family": "probabilistic", "cost": 1,
        "make": lambda: GaussianNB(),
        "space": {"model__var_smoothing": loguniform(1e-10, 1e-1)},
        "explain": "Applies Bayes' rule assuming every feature is independent and bell-shaped. Trains in milliseconds and works surprisingly well as a baseline, especially on text-like data.",
        "params": {"var_smoothing": "Adds a little variance to every feature for numerical stability."},
    },
    "tree": {
        "name": "Decision Tree", "family": "tree", "cost": 1,
        "make": lambda: DecisionTreeClassifier(random_state=RS),
        "space": {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        "explain": "Asks a chain of yes/no questions about the features. Fully interpretable and needs no scaling, but a single tree overfits easily. That is why we build forests out of them.",
        "params": {"max_depth": "How many questions deep the tree may go. Deeper = more overfitting.", "min_samples_leaf": "Minimum rows in a final answer bucket. Larger = smoother."},
    },
    "svm": {
        "name": "Support Vector Machine (RBF)", "family": "kernel", "cost": 4, "slow": True,
        "make": lambda: SVC(probability=False, random_state=RS),
        "space": {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-4, 1e0)},
        "explain": "Finds the boundary with the widest margin between classes, and the RBF kernel lets that boundary curve. Very strong on small and medium data. Training time grows roughly with the square of the row count, so it is skipped on large datasets.",
        "params": {"C": "Penalty for misclassified points. High C = fit training data harder.", "gamma": "How far one row's influence reaches. High gamma = wiggly boundary."},
    },
    "lda": {
        "name": "Linear Discriminant Analysis", "family": "probabilistic", "cost": 1,
        "make": lambda: LinearDiscriminantAnalysis(),
        "space": {},
        "explain": "Models each class as a Gaussian with a shared covariance and draws linear boundaries between them. Cheap, and doubles as a supervised dimensionality reducer.",
        "params": {},
    },
    "qda": {
        "name": "Quadratic Discriminant Analysis", "family": "probabilistic", "cost": 1,
        "make": lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        "space": {"model__reg_param": uniform(0, 0.5)},
        "explain": "Like LDA but each class gets its own covariance, so boundaries can curve. Needs more rows per class to estimate those covariances.",
        "params": {"reg_param": "Shrinks covariances toward identity for stability."},
    },
    "mlp": {
        "name": "Neural Network (MLP)", "family": "neural", "cost": 3, "slow": True,
        "make": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RS),
        "space": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)},
        "explain": "A small multilayer perceptron: layers of weighted sums and nonlinear activations trained by backpropagation. Flexible, needs scaling and more data than the others, and is the least interpretable.",
        "params": {"hidden_layer_sizes": "Neurons per hidden layer.", "alpha": "L2 regularization strength.", "learning_rate_init": "Step size for gradient descent."},
    },
    "rf": {
        "name": "Random Forest", "family": "bagging", "cost": 2,
        "make": lambda: RandomForestClassifier(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__max_features": ["sqrt", "log2", 0.5], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Bagging: trains many decision trees, each on a random sample of rows and a random subset of features, then averages their votes. Averaging cancels the overfitting of single trees. The default strong choice for tabular data.",
        "params": {"n_estimators": "Number of trees. More = better up to a point, slower.", "max_features": "Features each split may consider. Lower = more diverse trees."},
    },
    "et": {
        "name": "Extra Trees", "family": "bagging", "cost": 2,
        "make": lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Like Random Forest but split thresholds are chosen at random instead of searched. Even more variance reduction and faster to train, sometimes slightly less accurate.",
        "params": {},
    },
    "ada": {
        "name": "AdaBoost", "family": "boosting", "cost": 2,
        "make": lambda: AdaBoostClassifier(n_estimators=100, random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 1e0)},
        "explain": "Boosting: trains weak learners (stumps) one after another, each paying more attention to the rows the previous ones got wrong. The original boosting algorithm.",
        "params": {"learning_rate": "How much each new stump counts. Lower = needs more stumps."},
    },
    "gb": {
        "name": "Gradient Boosting", "family": "boosting", "cost": 3,
        "make": lambda: GradientBoostingClassifier(random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 3e-1), "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)},
        "explain": "Boosting with gradient descent: every new tree fits the errors (gradients) of the ensemble so far. Usually the most accurate classic model on tabular data, but slower and easier to overfit than a forest.",
        "params": {"learning_rate": "Shrinks each tree's contribution.", "n_estimators": "Number of boosting rounds.", "max_depth": "Depth of each small tree."},
    },
    "hgb": {
        "name": "Hist Gradient Boosting", "family": "boosting", "cost": 2,
        "make": lambda: HistGradientBoostingClassifier(random_state=RS),
        "space": {"model__learning_rate": loguniform(1e-2, 3e-1), "model__max_iter": randint(50, 300), "model__max_leaf_nodes": randint(15, 63), "model__l2_regularization": loguniform(1e-3, 1e1)},
        "explain": "Gradient boosting that first bins features into 255 buckets, the trick behind LightGBM and XGBoost. Much faster on large data and handles missing values natively.",
        "params": {"max_iter": "Boosting rounds.", "max_leaf_nodes": "Tree size per round."},
    },
}

REGRESSORS: dict[str, dict[str, Any]] = {
    "linear": {
        "name": "Linear Regression", "family": "linear", "cost": 1,
        "make": lambda: LinearRegression(),
        "space": {},
        "explain": "Fits the straight line (or hyperplane) that minimizes squared error. The baseline every regression should beat. No knobs to tune.",
        "params": {},
    },
    "ridge": {
        "name": "Ridge Regression", "family": "linear", "cost": 1,
        "make": lambda: Ridge(random_state=RS),
        "space": {"model__alpha": loguniform(1e-3, 1e3)},
        "explain": "Linear regression plus an L2 penalty that shrinks coefficients toward zero. Controls overfitting when features are many or correlated.",
        "params": {"alpha": "Penalty strength. Higher = smaller coefficients."},
    },
    "lasso": {
        "name": "Lasso Regression", "family": "linear", "cost": 1,
        "make": lambda: Lasso(random_state=RS, max_iter=5000),
        "space": {"model__alpha": loguniform(1e-4, 1e1)},
        "explain": "Linear regression plus an L1 penalty that drives some coefficients exactly to zero. Acts as automatic feature selection.",
        "params": {"alpha": "Penalty strength. Higher = more features dropped."},
    },
    "elastic": {
        "name": "ElasticNet", "family": "linear", "cost": 1,
        "make": lambda: ElasticNet(random_state=RS, max_iter=5000),
        "space": {"model__alpha": loguniform(1e-4, 1e1), "model__l1_ratio": uniform(0.1, 0.8)},
        "explain": "Mixes Ridge and Lasso penalties. Useful when groups of features are correlated and you still want some selection.",
        "params": {"l1_ratio": "0 = pure Ridge, 1 = pure Lasso."},
    },
    "knn": {
        "name": "KNN Regressor", "family": "distance", "cost": 2, "slow": True,
        "make": lambda: KNeighborsRegressor(n_neighbors=5),
        "space": {"model__n_neighbors": randint(3, 31), "model__weights": ["uniform", "distance"]},
        "explain": "Predicts the average target of the k nearest training rows. Simple, nonlinear, needs scaling, slow on large data.",
        "params": {"n_neighbors": "How many neighbours are averaged."},
    },
    "tree": {
        "name": "Decision Tree Regressor", "family": "tree", "cost": 1,
        "make": lambda: DecisionTreeRegressor(random_state=RS),
        "space": {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        "explain": "Splits the feature space into boxes and predicts the mean inside each box. Produces a step-shaped prediction and overfits on its own.",
        "params": {"max_depth": "Depth of the tree."},
    },
    "svr": {
        "name": "Support Vector Regression", "family": "kernel", "cost": 4, "slow": True,
        "make": lambda: SVR(),
        "space": {"model__C": loguniform(1e-1, 1e2), "model__epsilon": loguniform(1e-2, 1e0), "model__gamma": loguniform(1e-4, 1e0)},
        "explain": "SVM for numbers: ignores errors smaller than epsilon and penalizes the rest. Strong on small data, too slow for large data.",
        "params": {"epsilon": "Width of the no-penalty tube around the prediction."},
    },
    "mlp": {
        "name": "Neural Network (MLP)", "family": "neural", "cost": 3, "slow": True,
        "make": lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=RS),
        "space": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)},
        "explain": "A small neural network for regression. Needs scaled features and a scaled target to train well.",
        "params": {},
    },
    "rf": {
        "name": "Random Forest Regressor", "family": "bagging", "cost": 2,
        "make": lambda: RandomForestRegressor(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__max_features": ["sqrt", "log2", 0.5, 1.0], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Bagging of regression trees. Averages many noisy trees into a smooth, robust predictor. Rarely the worst choice.",
        "params": {},
    },
    "et": {
        "name": "Extra Trees Regressor", "family": "bagging", "cost": 2,
        "make": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Random Forest with random split thresholds. Faster, slightly smoother.",
        "params": {},
    },
    "ada": {
        "name": "AdaBoost Regressor", "family": "boosting", "cost": 2,
        "make": lambda: AdaBoostRegressor(n_estimators=100, random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 1e0), "model__loss": ["linear", "square", "exponential"]},
        "explain": "Sequential boosting of shallow trees, reweighting rows with large errors.",
        "params": {},
    },
    "gb": {
        "name": "Gradient Boosting Regressor", "family": "boosting", "cost": 3,
        "make": lambda: GradientBoostingRegressor(random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 3e-1), "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)},
        "explain": "Each new tree fits the residuals of the previous ensemble. Typically the most accurate classic choice for tabular regression.",
        "params": {},
    },
    "hgb": {
        "name": "Hist Gradient Boosting Regressor", "family": "boosting", "cost": 2,
        "make": lambda: HistGradientBoostingRegressor(random_state=RS),
        "space": {"model__learning_rate": loguniform(1e-2, 3e-1), "model__max_iter": randint(50, 300), "model__max_leaf_nodes": randint(15, 63), "model__l2_regularization": loguniform(1e-3, 1e1)},
        "explain": "Histogram-based gradient boosting, the fast variant used by LightGBM. Handles missing values natively.",
        "params": {},
    },
}

FAMILY_TEXT = {
    "linear": "Linear models: a weighted sum of the features. Fast, interpretable, need scaling, cannot bend.",
    "distance": "Distance-based: predictions come from the nearest training rows. No real training step, needs scaling.",
    "probabilistic": "Probabilistic: model each class with a distribution and apply Bayes' rule.",
    "tree": "Single decision tree: a flowchart of yes/no questions. Interpretable, overfits alone.",
    "kernel": "Kernel methods: project data into a richer space where a linear boundary works. Strong on small data, slow on large.",
    "neural": "Neural networks: stacked layers of weighted sums and activations trained by backpropagation.",
    "bagging": "Bagging ensembles: many models trained on random samples in parallel, then averaged. Reduces variance.",
    "boosting": "Boosting ensembles: models trained one after another, each correcting the last. Reduces bias.",
}


def registry(task: str) -> dict[str, dict[str, Any]]:
    if task == "classification":
        return CLASSIFIERS
    if task == "regression":
        return REGRESSORS
    raise ValueError(task)


def public_catalog() -> dict:
    def strip(reg: dict) -> list[dict]:
        out = []
        for key, m in reg.items():
            out.append({
                "key": key, "name": m["name"], "family": m["family"], "cost": m["cost"],
                "slow": bool(m.get("slow")), "explain": m["explain"], "params": m["params"],
                "tunable": sorted(k.replace("model__", "") for k in m["space"].keys()),
            })
        return out
    return {
        "classification": strip(CLASSIFIERS),
        "regression": strip(REGRESSORS),
        "families": FAMILY_TEXT,
        "ensembles": ENSEMBLE_TEXT,
        "unsupervised": UNSUPERVISED_TEXT,
    }


ENSEMBLE_TEXT = {
    "voting": "Voting: several different models predict, and the ensemble takes the majority vote (hard) or averages the probabilities (soft). Works best when the members make different kinds of mistakes.",
    "stacking": "Stacking: base models predict, and a final model (the meta-learner) learns how much to trust each one. Base predictions are produced with cross-validation so the meta-learner never sees leaked fits.",
    "bagging": "Bagging: one model type, many copies, each trained on a bootstrap sample of the rows. Random Forest is bagging of trees plus random feature subsets.",
    "boosting": "Boosting: one weak model type trained in sequence, each round focusing on what the previous rounds got wrong. AdaBoost reweights rows; Gradient Boosting fits residuals.",
}

UNSUPERVISED_TEXT = {
    "kmeans": "K-Means: picks k centres and assigns every row to the nearest one, then moves the centres to the mean of their rows, repeating until stable. You must choose k; the elbow and silhouette plots help.",
    "dbscan": "DBSCAN: grows clusters from dense regions. Finds any cluster shape and labels sparse points as noise (-1). No k needed, but eps and min_samples matter a lot.",
    "agglomerative": "Agglomerative clustering: starts with every row as its own cluster and merges the closest pair repeatedly. The merge history forms a dendrogram.",
    "gmm": "Gaussian Mixture: like soft K-Means where each cluster is a Gaussian blob with its own shape and each row gets a membership probability.",
    "pca": "PCA: rotates the data onto new axes ordered by how much variance they explain, then keeps the first few. Lossy compression that keeps the most structure.",
    "tsne": "t-SNE: a 2D map that keeps close points close. Great for looking, useless for distances between far clusters. Capped to a few thousand rows because it is slow.",
    "isoforest": "Isolation Forest: anomalies are the rows that random splits isolate quickly. A fast, general outlier detector.",
    "lof": "Local Outlier Factor: a row is an outlier if it is much less dense than its neighbours.",
}
