import numpy as np
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from pyswarm import pso
from src.config import MODEL_PKL_PATH
from src.models.feature_selection.anfis import calculate_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def common_train(X_train, X_test, y_train, y_test, feature_indices, model, model_type, clf):
    def objective_function(params):
        threshold = params[0]
        ids = calculate_id(feature_indices, params[1:])
        selected_indices = np.where(ids > threshold)[0]
        if len(selected_indices) == 0:
            return 1e6

        selected_features = X_train.columns[selected_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(scaler.transform(X_test[selected_features]))

        return mean_squared_error(y_test, y_pred)

    lb = [0.5] + [0.1] * (len(feature_indices) - 1)
    ub = [0.9] + [1.0] * (len(feature_indices) - 1)
    optimized_params, _ = pso(objective_function, lb, ub, swarmsize=20, maxiter=10, debug=False)

    ids = calculate_id(feature_indices, optimized_params)
    selected_features = X_train.columns[np.where(ids > optimized_params[0])[0]]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    clf.fit(X_train_scaled, y_train)

    with open(f"{MODEL_PKL_PATH}/{model_type}/{model}.pkl", "wb") as f:
        pickle.dump(clf, f)
    np.save(f"{MODEL_PKL_PATH}/{model_type}/{model}_feat.npy", selected_features)

    y_pred = clf.predict(X_test_scaled)
    roc_auc = 0
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    logging.info(f"{model} trained with ROC AUC: {roc_auc:.2f}")
    logging.info(classification_report(y_test, y_pred))

