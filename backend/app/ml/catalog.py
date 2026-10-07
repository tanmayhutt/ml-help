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
        "explain": "Draws a straight line between the groups and gives a probability for each guess. Fast and reliable. Works best when the groups can be separated by a straight line.",
        "params": {"C": "Inverse regularization. Small C = simpler, more regularized model."},
    },
    "knn": {
        "name": "K-Nearest Neighbors", "family": "distance", "cost": 2, "slow": True,
        "make": lambda: KNeighborsClassifier(n_neighbors=5),
        "space": {"model__n_neighbors": randint(3, 31), "model__weights": ["uniform", "distance"]},
        "explain": "Looks at the k most similar rows it has seen and copies the majority answer. Simple and often good, but slow on big files.",
        "params": {"n_neighbors": "How many neighbours vote. Small k = jagged boundary, large k = smooth."},
    },
    "nb": {
        "name": "Gaussian Naive Bayes", "family": "probabilistic", "cost": 1,
        "make": lambda: GaussianNB(),
        "space": {"model__var_smoothing": loguniform(1e-10, 1e-1)},
        "explain": "Uses simple probability rules, treating each column on its own. Trains instantly and is a good baseline.",
        "params": {"var_smoothing": "Adds a little variance to every feature for numerical stability."},
    },
    "tree": {
        "name": "Decision Tree", "family": "tree", "cost": 1,
        "make": lambda: DecisionTreeClassifier(random_state=RS),
        "space": {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        "explain": "Asks a chain of yes/no questions about the columns. Easy to read, but one tree alone tends to memorise the data.",
        "params": {"max_depth": "How many questions deep the tree may go. Deeper = more overfitting.", "min_samples_leaf": "Minimum rows in a final answer bucket. Larger = smoother."},
    },
    "svm": {
        "name": "Support Vector Machine (RBF)", "family": "kernel", "cost": 4, "slow": True,
        "make": lambda: SVC(probability=False, random_state=RS),
        "space": {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-4, 1e0)},
        "explain": "Finds the boundary with the biggest gap between groups, and can bend that boundary. Very strong on small and medium files, too slow on big ones.",
        "params": {"C": "Penalty for misclassified points. High C = fit training data harder.", "gamma": "How far one row's influence reaches. High gamma = wiggly boundary."},
    },
    "lda": {
        "name": "Linear Discriminant Analysis", "family": "probabilistic", "cost": 1,
        "make": lambda: LinearDiscriminantAnalysis(),
        "space": {},
        "explain": "Describes each group as a bell-shaped cloud and draws straight lines between the clouds. Very fast.",
        "params": {},
    },
    "qda": {
        "name": "Quadratic Discriminant Analysis", "family": "probabilistic", "cost": 1,
        "make": lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
        "space": {"model__reg_param": uniform(0, 0.5)},
        "explain": "Like LDA, but each group gets its own cloud shape, so the boundaries can curve. Needs more rows per group.",
        "params": {"reg_param": "Shrinks covariances toward identity for stability."},
    },
    "mlp": {
        "name": "Neural Network (MLP)", "family": "neural", "cost": 3, "slow": True,
        "make": lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RS),
        "space": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)},
        "explain": "A small neural network. Flexible, but needs more rows than the others and is the hardest to explain.",
        "params": {"hidden_layer_sizes": "Neurons per hidden layer.", "alpha": "L2 regularization strength.", "learning_rate_init": "Step size for gradient descent."},
    },
    "rf": {
        "name": "Random Forest", "family": "bagging", "cost": 2,
        "make": lambda: RandomForestClassifier(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__max_features": ["sqrt", "log2", 0.5], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Builds many decision trees on random slices of the data and lets them vote. Averaging many trees cancels out their mistakes. A strong default for tables.",
        "params": {"n_estimators": "Number of trees. More = better up to a point, slower.", "max_features": "Features each split may consider. Lower = more diverse trees."},
    },
    "et": {
        "name": "Extra Trees", "family": "bagging", "cost": 2,
        "make": lambda: ExtraTreesClassifier(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Like Random Forest but with more randomness in each tree. Faster to train, sometimes slightly less accurate.",
        "params": {},
    },
    "ada": {
        "name": "AdaBoost", "family": "boosting", "cost": 2,
        "make": lambda: AdaBoostClassifier(n_estimators=100, random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 1e0)},
        "explain": "Trains many tiny trees one after another, each focusing on the rows the previous ones got wrong.",
        "params": {"learning_rate": "How much each new stump counts. Lower = needs more stumps."},
    },
    "gb": {
        "name": "Gradient Boosting", "family": "boosting", "cost": 3,
        "make": lambda: GradientBoostingClassifier(random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 3e-1), "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)},
        "explain": "Trains trees one after another, each one correcting the errors of the ones before. Often the most accurate choice for tables, but slower than a forest.",
        "params": {"learning_rate": "Shrinks each tree's contribution.", "n_estimators": "Number of boosting rounds.", "max_depth": "Depth of each small tree."},
    },
    "hgb": {
        "name": "Hist Gradient Boosting", "family": "boosting", "cost": 2,
        "make": lambda: HistGradientBoostingClassifier(random_state=RS),
        "space": {"model__learning_rate": loguniform(1e-2, 3e-1), "model__max_iter": randint(50, 300), "model__max_leaf_nodes": randint(15, 63), "model__l2_regularization": loguniform(1e-3, 1e1)},
        "explain": "A faster version of Gradient Boosting that works well on big files and handles blanks on its own.",
        "params": {"max_iter": "Boosting rounds.", "max_leaf_nodes": "Tree size per round."},
    },
}

REGRESSORS: dict[str, dict[str, Any]] = {
    "linear": {
        "name": "Linear Regression", "family": "linear", "cost": 1,
        "make": lambda: LinearRegression(),
        "space": {},
        "explain": "Fits the best straight line through the data. The simplest model and the baseline every other model should beat.",
        "params": {},
    },
    "ridge": {
        "name": "Ridge Regression", "family": "linear", "cost": 1,
        "make": lambda: Ridge(random_state=RS),
        "space": {"model__alpha": loguniform(1e-3, 1e3)},
        "explain": "A straight-line model that is held back from relying too much on any one column. Helps when columns overlap or are many.",
        "params": {"alpha": "Penalty strength. Higher = smaller coefficients."},
    },
    "lasso": {
        "name": "Lasso Regression", "family": "linear", "cost": 1,
        "make": lambda: Lasso(random_state=RS, max_iter=5000),
        "space": {"model__alpha": loguniform(1e-4, 1e1)},
        "explain": "A straight-line model that switches off columns it does not need. Useful for finding which columns matter.",
        "params": {"alpha": "Penalty strength. Higher = more features dropped."},
    },
    "elastic": {
        "name": "ElasticNet", "family": "linear", "cost": 1,
        "make": lambda: ElasticNet(random_state=RS, max_iter=5000),
        "space": {"model__alpha": loguniform(1e-4, 1e1), "model__l1_ratio": uniform(0.1, 0.8)},
        "explain": "A mix of Ridge and Lasso.",
        "params": {"l1_ratio": "0 = pure Ridge, 1 = pure Lasso."},
    },
    "knn": {
        "name": "KNN Regressor", "family": "distance", "cost": 2, "slow": True,
        "make": lambda: KNeighborsRegressor(n_neighbors=5),
        "space": {"model__n_neighbors": randint(3, 31), "model__weights": ["uniform", "distance"]},
        "explain": "Finds the k most similar rows and averages their values. Simple, but slow on big files.",
        "params": {"n_neighbors": "How many neighbours are averaged."},
    },
    "tree": {
        "name": "Decision Tree Regressor", "family": "tree", "cost": 1,
        "make": lambda: DecisionTreeRegressor(random_state=RS),
        "space": {"model__max_depth": randint(2, 20), "model__min_samples_leaf": randint(1, 20)},
        "explain": "Asks yes/no questions to sort rows into boxes, then predicts the average of each box. Tends to memorise on its own.",
        "params": {"max_depth": "Depth of the tree."},
    },
    "svr": {
        "name": "Support Vector Regression", "family": "kernel", "cost": 4, "slow": True,
        "make": lambda: SVR(),
        "space": {"model__C": loguniform(1e-1, 1e2), "model__epsilon": loguniform(1e-2, 1e0), "model__gamma": loguniform(1e-4, 1e0)},
        "explain": "A flexible curve fitter that ignores small errors. Strong on small files, too slow on big ones.",
        "params": {"epsilon": "Width of the no-penalty tube around the prediction."},
    },
    "mlp": {
        "name": "Neural Network (MLP)", "family": "neural", "cost": 3, "slow": True,
        "make": lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=RS),
        "space": {"model__hidden_layer_sizes": [(32,), (64,), (64, 32)], "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(1e-4, 1e-2)},
        "explain": "A small neural network for predicting numbers. Needs plenty of rows.",
        "params": {},
    },
    "rf": {
        "name": "Random Forest Regressor", "family": "bagging", "cost": 2,
        "make": lambda: RandomForestRegressor(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__max_features": ["sqrt", "log2", 0.5, 1.0], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Many trees on random slices of the data, averaged. Robust and rarely a bad choice.",
        "params": {},
    },
    "et": {
        "name": "Extra Trees Regressor", "family": "bagging", "cost": 2,
        "make": lambda: ExtraTreesRegressor(n_estimators=200, n_jobs=_nj(), random_state=RS),
        "space": {"model__n_estimators": randint(100, 400), "model__max_depth": [None, 5, 10, 20], "model__min_samples_leaf": randint(1, 10)},
        "explain": "Random Forest with extra randomness. Faster, slightly smoother.",
        "params": {},
    },
    "ada": {
        "name": "AdaBoost Regressor", "family": "boosting", "cost": 2,
        "make": lambda: AdaBoostRegressor(n_estimators=100, random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 1e0), "model__loss": ["linear", "square", "exponential"]},
        "explain": "Small trees trained one after another, each focusing on the rows with the biggest errors.",
        "params": {},
    },
    "gb": {
        "name": "Gradient Boosting Regressor", "family": "boosting", "cost": 3,
        "make": lambda: GradientBoostingRegressor(random_state=RS),
        "space": {"model__n_estimators": randint(50, 300), "model__learning_rate": loguniform(1e-2, 3e-1), "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)},
        "explain": "Trees trained one after another, each correcting the previous errors. Often the most accurate choice for tables.",
        "params": {},
    },
    "hgb": {
        "name": "Hist Gradient Boosting Regressor", "family": "boosting", "cost": 2,
        "make": lambda: HistGradientBoostingRegressor(random_state=RS),
        "space": {"model__learning_rate": loguniform(1e-2, 3e-1), "model__max_iter": randint(50, 300), "model__max_leaf_nodes": randint(15, 63), "model__l2_regularization": loguniform(1e-3, 1e1)},
        "explain": "A faster Gradient Boosting that works well on big files and handles blanks on its own.",
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


BALANCEABLE = {"logreg", "svm", "tree", "rf", "et", "hgb"}


def set_balanced(est, key: str):
    """Give minority classes more weight where the model supports it."""
    if key in BALANCEABLE and "class_weight" in est.get_params():
        est.set_params(class_weight="balanced")
    return est


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
    "voting": "Several different models each make a guess, and the final answer is the majority vote or the average of their confidence. Works best when the models make different kinds of mistakes.",
    "stacking": "Several models make guesses, and one final model learns how much to trust each of them.",
    "bagging": "Many copies of one model, each trained on a random slice of the rows, then averaged. Random Forest is bagging of decision trees.",
    "boosting": "One model type trained many times in a row, each round focusing on the rows the previous rounds got wrong.",
}

UNSUPERVISED_TEXT = {
    "kmeans": "K-Means picks a centre for each group and puts every row with its nearest centre, then moves the centres and repeats until nothing changes. You choose how many groups.",
    "dbscan": "DBSCAN grows groups out of crowded areas. It can find groups of any shape, and rows that sit alone are marked as 'no group'.",
    "agglomerative": "Starts with every row on its own and keeps merging the two closest groups until the number you asked for is left.",
    "gmm": "Like K-Means, but each group is a soft, stretchable blob and every row gets a probability of belonging to each group.",
    "pca": "Squeezes all the columns into two new ones that keep as much of the differences between rows as possible. Fast and good for a first look.",
    "tsne": "Draws a 2D map where similar rows end up close together. Good for spotting tight groups, slow on big files.",
    "isoforest": "Rows that are easy to separate from the rest with a few random cuts are the unusual ones. Fast and works on most data.",
    "lof": "A row is unusual if it sits in a much emptier spot than its neighbours.",
}
