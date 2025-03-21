import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyswarm import pso

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_importance_degree(params, indices_df):
    weighted_scores = (
        indices_df['Fisher_Index'] * params[0] +
        indices_df['Correlation_Index'] * params[1] +
        indices_df['T-test_Index'] * params[2] +
        indices_df['Mutual_Information_Index'] * params[3]
    )
    return np.where(weighted_scores > 0.75, 1, np.where(weighted_scores > 0.4, 0.5, 0))


def select_features(features_df, importance_degree):
    return features_df.loc[:, importance_degree == 1]


def optimize_svm_anfis(X_train, y_train, X_val, y_val, indices_df):
    def objective(params):
        importance_degree = calculate_importance_degree(params[:4], indices_df)
        X_train_sel = select_features(X_train, importance_degree)
        X_val_sel = select_features(X_val, importance_degree)

        if X_train_sel.shape[1] == 0:
            return 1.0

        model = SVC(kernel='rbf', C=params[4], gamma=params[5])
        model.fit(X_train_sel, y_train)
        accuracy = accuracy_score(y_val, model.predict(X_val_sel))
        return -accuracy

    lb = [0, 0, 0, 0, 0.1, 0.001]
    ub = [1, 1, 1, 1, 10, 1]

    optimal_params, _ = pso(objective, lb, ub, swarmsize=3, maxiter=3)
    return optimal_params


def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)
    f1 = f1_score(y, y_pred, average=None)
    conf_matrix = confusion_matrix(y, y_pred)

    logging.info(f"{dataset_name} Accuracy: {accuracy}")
    logging.info(f"{dataset_name} Precision: {precision}")
    logging.info(f"{dataset_name} Recall: {recall}")
    logging.info(f"{dataset_name} F1 Score: {f1}")
    logging.info(f"{dataset_name} Confusion Matrix:\n{conf_matrix}")


# Main Training Pipeline
def SvmA_train(X_train, X_val, y_train, y_val, indices_df, model):
    logging.info("Starting SVM-ANFIS optimization...")

    optimal_params = optimize_svm_anfis(X_train, y_train, X_val, y_val, indices_df)
    best_anfis_params, best_C, best_gamma = optimal_params[:4], optimal_params[4], optimal_params[5]

    importance_degree = calculate_importance_degree(best_anfis_params, indices_df)
    X_train_sel = select_features(X_train, importance_degree)
    X_val_sel = select_features(X_val, importance_degree)

    svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    svm_final.fit(X_train_sel, y_train)

    joblib.dump(svm_final, f'model/{model}/svm_model_final.pkl')
    joblib.dump(X_train_sel, f'model/{model}/selected_features_train.pkl')

    logging.info("Model and features saved successfully.")

    evaluate_model(svm_final, X_train_sel, y_train, "Training")
    evaluate_model(svm_final, X_val_sel, y_val, "Validation")

    logging.info(f"Optimal ANFIS Parameters: {best_anfis_params}")
    logging.info(f"Optimal SVM Parameters (C, gamma): ({best_C}, {best_gamma})")


