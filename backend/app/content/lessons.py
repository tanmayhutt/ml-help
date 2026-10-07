"""Guided lessons: each one is a sample dataset plus a sequence of jobs with a question to answer."""
LESSONS = [
    {
        "key": "scaling",
        "title": "Why scaling matters",
        "sample": "wine",
        "intro": "Wine features range from 0.1 to 1500. Distance-based models are fooled by the big columns unless we scale.",
        "steps": [
            {"text": "Race all models with no scaling. Watch KNN, SVM and Logistic Regression.", "job": {"kind": "leaderboard", "params": {"task": "classification", "target": "cultivar", "models": ["knn", "svm", "logreg", "rf", "tree"], "preprocess": {"scaler": "none"}, "include_slow": True}}},
            {"text": "Race again with standard scaling. Trees barely move; distance models jump.", "job": {"kind": "leaderboard", "params": {"task": "classification", "target": "cultivar", "models": ["knn", "svm", "logreg", "rf", "tree"], "preprocess": {"scaler": "standard"}, "include_slow": True}}},
        ],
        "takeaway": "Scaling changes nothing about a tree, which only asks 'is x greater than t'. It changes everything for a model that measures distances or sums weighted features.",
    },
    {
        "key": "overfit",
        "title": "Overfitting a decision tree",
        "sample": "breast_cancer",
        "intro": "A tree with no depth limit memorizes the training rows. Compare a deep tree, a shallow tree, and a forest.",
        "steps": [
            {"text": "Train an unlimited-depth tree. Note the train score of 1.0 and the gap to the test score.", "job": {"kind": "train", "params": {"task": "classification", "target": "diagnosis", "spec": {"kind": "single", "model": "tree", "params": {"max_depth": None}}}}},
            {"text": "Train a tree limited to depth 3. The train score drops, the gap shrinks.", "job": {"kind": "train", "params": {"task": "classification", "target": "diagnosis", "spec": {"kind": "single", "model": "tree", "params": {"max_depth": 3}}}}},
            {"text": "Train a Random Forest. Many overfit trees averaged together generalize better than one.", "job": {"kind": "train", "params": {"task": "classification", "target": "diagnosis", "spec": {"kind": "single", "model": "rf"}}}},
        ],
        "takeaway": "Overfitting shows up as a train/test gap. You can fix it by constraining the model (depth) or by averaging many models (bagging).",
    },
    {
        "key": "ensembles",
        "title": "Bagging versus boosting versus stacking",
        "sample": "digits",
        "intro": "Three ways to combine models, on the same data.",
        "steps": [
            {"text": "Bagging: 30 bootstrapped decision trees.", "job": {"kind": "train", "params": {"task": "classification", "target": "digit", "spec": {"kind": "bagging", "model": "tree", "n_estimators": 30}}}},
            {"text": "Boosting: Hist Gradient Boosting, trees in sequence.", "job": {"kind": "train", "params": {"task": "classification", "target": "digit", "spec": {"kind": "single", "model": "hgb"}}}},
            {"text": "Stacking: Logistic Regression, KNN and Random Forest with a logistic meta-learner.", "job": {"kind": "train", "params": {"task": "classification", "target": "digit", "spec": {"kind": "stacking", "members": ["logreg", "knn", "rf"], "final": "logreg"}}}},
            {"text": "Soft voting over the same three members.", "job": {"kind": "train", "params": {"task": "classification", "target": "digit", "spec": {"kind": "voting", "members": ["logreg", "knn", "rf"], "voting": "soft"}}}},
        ],
        "takeaway": "Bagging lowers variance, boosting lowers bias, stacking and voting exploit diversity between different model families. On most data the differences are small; on hard data they are not.",
    },
    {
        "key": "regression",
        "title": "Linear versus nonlinear regression",
        "sample": "diabetes",
        "intro": "A hard regression problem where the best R2 is only around 0.5.",
        "steps": [
            {"text": "Race everything. Note how close linear models and ensembles are.", "job": {"kind": "leaderboard", "params": {"task": "regression", "target": "progression"}}},
            {"text": "Tune Ridge's alpha. Regularization helps when features are correlated.", "job": {"kind": "tune", "params": {"task": "regression", "target": "progression", "model": "ridge", "n_iter": 12}}},
            {"text": "Learning curve for Gradient Boosting: are we data-limited?", "job": {"kind": "curve", "params": {"task": "regression", "target": "progression", "model": "gb"}}},
        ],
        "takeaway": "A low ceiling across all models means the features do not contain more signal. No amount of model tuning fixes missing information.",
    },
    {
        "key": "clustering",
        "title": "Finding groups without labels",
        "sample": "iris",
        "intro": "Pretend we do not know the species. Can clustering rediscover them?",
        "steps": [
            {"text": "KMeans with k=3. Check the silhouette sweep: does the data agree that 3 is right?", "job": {"kind": "cluster", "params": {"algorithm": "kmeans", "k": 3, "preprocess": {"drop_columns": ["species"]}}}},
            {"text": "DBSCAN with eps=0.6. Two species overlap, so density finds only two groups plus noise.", "job": {"kind": "cluster", "params": {"algorithm": "dbscan", "eps": 0.6, "min_samples": 5, "preprocess": {"drop_columns": ["species"]}}}},
            {"text": "PCA coloured by the real species. Two components explain most of the variance.", "job": {"kind": "reduce", "params": {"algorithm": "pca", "n_components": 2, "color_by": "species", "preprocess": {"drop_columns": ["species"]}}}},
        ],
        "takeaway": "Clustering finds structure, not labels. Whether the clusters match human categories depends on whether those categories are what separates the data geometrically.",
    },
]
