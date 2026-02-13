"""Classifier selector utility for traditional ML models.

Maps unified `model_name` strings to instantiated scikit-learn / boosting
classifier objects. Used by training pipeline after naming unification.
"""

import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier


def get_classifier(model_name: str, seed: int = 42):
    """
    Return a classifier instance based on the given model name.

    Supported model names include:
    - ``"RF"`` : Random Forest
    - ``"BalancedRF"`` : Balanced Random Forest
    - ``"SvmW"`` : Support Vector Machine with RBF kernel
    - ``"DecisionTree"`` : Decision Tree
    - ``"AdaBoost"`` : AdaBoost classifier
    - ``"GradientBoosting"`` : Gradient Boosting classifier
    - ``"XGBoost"`` : XGBoost classifier
    - ``"LightGBM"`` : LightGBM classifier
    - ``"CatBoost"`` : CatBoost classifier
    - ``"LogisticRegression"`` : Logistic Regression
    - ``"SVM"`` : Linear Support Vector Machine
    - ``"K-Nearest Neighbors"`` : KNN classifier
    - ``"MLP"`` : Multi-layer Perceptron classifier

    Parameters
    ----------

    model_name : str
        Identifier string for the classifier.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------

    sklearn.base.BaseEstimator
        Instantiated classifier object.

    Raises
    ------

    ValueError
        If the given ``model_name`` is not supported.
    """
    classifiers = {
        "RF": RandomForestClassifier(random_state=seed),
        "BalancedRF": BalancedRandomForestClassifier(random_state=seed),
        "EasyEnsemble": EasyEnsembleClassifier(random_state=seed),
        "SvmW": SVC(kernel="rbf", C=300.0, probability=True, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
        "AdaBoost": AdaBoostClassifier(random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=seed),
        "LightGBM": LGBMClassifier(random_state=seed),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "SVM": SVC(kernel="linear", probability=True, random_state=seed),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "MLP": MLPClassifier(max_iter=500, random_state=seed),
    }

    clf = classifiers.get(model_name)
    if clf is None:
        logging.error(f"Classifier '{model_name}' is not supported.")
        raise ValueError(f"Classifier '{model_name}' is not defined.")

    return clf

