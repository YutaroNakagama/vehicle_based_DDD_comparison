"""Classifier selector utility for traditional machine learning models.

Provides a mapping from string-based model names to instantiated scikit-learn
or boosting-based classifier objects.

Used in the model training pipeline to abstract classifier creation.
"""

import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

def get_classifier(model_name: str):
    """Return a classifier instance based on the given model name.

    Supported model names include:
    - 'RF': Random Forest
    - 'SvmW': Support Vector Machine with RBF kernel
    - 'XGBoost': XGBoost classifier
    - 'LightGBM': LightGBM classifier
    - 'LogisticRegression': Logistic regression

    Args:
        model_name (str): Identifier string for the classifier.

    Returns:
        sklearn.base.BaseEstimator: Instantiated classifier object.

    Raises:
        ValueError: If the given model_name is not supported.
    """
    classifiers = {
        "RF": RandomForestClassifier(random_state=42),
        "SvmW": SVC(kernel="rbf", probability=True, random_state=42),
        # Extendable section (commented alternatives for future use):
        # "Decision Tree": DecisionTreeClassifier(random_state=42),
        # "AdaBoost": AdaBoostClassifier(random_state=42),
        # "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        # "LightGBM": LGBMClassifier(random_state=42),
        # "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        # "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        # "SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=42),
        # "K-Nearest Neighbors": KNeighborsClassifier(),
        # "MLP (Neural Network)": MLPClassifier(max_iter=500, random_state=42),
    }

    clf = classifiers.get(model_name)
    if clf is None:
        logging.error(f"Classifier '{model_name}' is not supported.")
        raise ValueError(f"Classifier '{model_name}' is not defined.")

    return clf

