import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

def get_classifier(model_name):
    classifiers = {
        "RF": RandomForestClassifier(random_state=42),
        "SvmW": SVC(kernel="rbf", probability=True, random_state=42),
        #"Decision Tree": DecisionTreeClassifier(random_state=42),
        #"AdaBoost": AdaBoostClassifier(random_state=42),
        #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
        #"XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        #"LightGBM": lgb.LGBMClassifier(random_state=42),
        #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
        #"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        #"Perceptron": Perceptron(max_iter=1000, random_state=42),
        #"SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=42),
        #"SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        #"K-Nearest Neighbors": KNeighborsClassifier(),
        #"MLP (Neural Network)": MLPClassifier(max_iter=500, random_state=42),
    }

    clf = classifiers.get(model_name)
    if clf is None:
        logging.error(f"Classifier '{model_name}' is not supported.")
        raise ValueError(f"Classifier '{model_name}' is not defined.")
    
    return clf

