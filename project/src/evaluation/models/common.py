import pickle
import numpy as np
import logging
from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
from sklearn.preprocessing import StandardScaler
from src.config import MODEL_PKL_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def common_eval(X_train, X_test, y_train, y_test, model_name, model_type):
    model_path = f"{MODEL_PKL_PATH}/{model_type}/{model_name}.pkl"
    features_path = f"{MODEL_PKL_PATH}/{model_type}/{model_name}_feat.npy"

    with open(model_path, "rb") as f:
        clf = pickle.load(f)

    selected_features = np.load(features_path, allow_pickle=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    #clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    roc_auc = 0
    if hasattr(clf, "predict_proba"):
        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    mse = mean_squared_error(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"Model: {model_name}")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"ROC AUC: {roc_auc:.4f}")
    logging.info(f"Classification Report:\n{report}")

