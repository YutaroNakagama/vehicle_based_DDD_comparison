#import numpy as np
#import pickle
#import logging
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import classification_report, roc_curve, auc, mean_squared_error
#from pyswarm import pso
#from src.config import MODEL_PKL_PATH
#from src.models.feature_selection.anfis import calculate_id
#
#logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
#
#def common_train(X_train, X_test, y_train, y_test, feature_indices, model, model_type, clf):
#    def objective_function(params):
#        threshold = params[0]
#        ids = calculate_id(feature_indices, params[1:])
#        selected_indices = np.where(ids > threshold)[0]
#        if len(selected_indices) == 0:
#            return 1e6
#
#        selected_features = X_train.columns[selected_indices]
#        scaler = StandardScaler()
#        X_train_scaled = scaler.fit_transform(X_train[selected_features])
#        clf.fit(X_train_scaled, y_train)
#        y_pred = clf.predict(scaler.transform(X_test[selected_features]))
#
#        return mean_squared_error(y_test, y_pred)
#
#    lb = [0.5] + [0.1] * (len(feature_indices) - 1)
#    ub = [0.9] + [1.0] * (len(feature_indices) - 1)
#    optimized_params, _ = pso(objective_function, lb, ub, swarmsize=20, maxiter=10, debug=False)
#
#    ids = calculate_id(feature_indices, optimized_params)
#    selected_features = X_train.columns[np.where(ids > optimized_params[0])[0]]
#
#    scaler = StandardScaler()
#    X_train_scaled = scaler.fit_transform(X_train[selected_features])
#    X_test_scaled = scaler.transform(X_test[selected_features])
#
#    clf.fit(X_train_scaled, y_train)
#
#    with open(f"{MODEL_PKL_PATH}/{model_type}/{model}.pkl", "wb") as f:
#        pickle.dump(clf, f)
#    np.save(f"{MODEL_PKL_PATH}/{model_type}/{model}_feat.npy", selected_features)
#
#    y_pred = clf.predict(X_test_scaled)
#    roc_auc = 0
#    if hasattr(clf, "predict_proba"):
#        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
#        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
#        roc_auc = auc(fpr, tpr)
#
#    logging.info(f"{model} trained with ROC AUC: {roc_auc:.2f}")
#    logging.info(classification_report(y_test, y_pred))
#

import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def common_train(X_train, X_test, y_train, y_test, feature_indices, model, model_type, clf):
    from src.models.feature_selection.anfis import calculate_id
    from sklearn.preprocessing import StandardScaler
    import pickle
    import numpy as np
    import logging
    from sklearn.metrics import classification_report, roc_curve, auc
    from src.config import MODEL_PKL_PATH

    def objective(trial):
        threshold = trial.suggest_float("threshold", 0.5, 0.9)
        ids = calculate_id(feature_indices, [threshold] + [1.0] * (len(feature_indices.columns) * 2))
        selected_indices = np.where(ids > threshold)[0]

        if len(selected_indices) == 0:
            return 1.0  # high error

        selected_features = X_train.columns[selected_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train[selected_features])
        X_val_scaled = scaler.transform(X_test[selected_features])

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "random_state": 42,
        }

        clf = RandomForestClassifier(**params)
        score = cross_val_score(clf, X_train_scaled, y_train, cv=3, scoring="accuracy").mean()
        return -score  # maximize accuracy

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_threshold = best_params.pop("threshold")

    ids = calculate_id(feature_indices, [best_threshold] + [1.0] * (len(feature_indices.columns) * 2))
    selected_features = X_train.columns[np.where(ids > best_threshold)[0]]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[selected_features])
    X_test_scaled = scaler.transform(X_test[selected_features])

    best_clf = RandomForestClassifier(**best_params)
    best_clf.fit(X_train_scaled, y_train)

    # 保存
    with open(f"{MODEL_PKL_PATH}/{model_type}/{model}.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    np.save(f"{MODEL_PKL_PATH}/{model_type}/{model}_feat.npy", selected_features)

    # 評価
    y_pred = best_clf.predict(X_test_scaled)
    roc_auc = 0
    if hasattr(best_clf, "predict_proba"):
        y_pred_proba = best_clf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    logging.info(f"{model} (Optuna) trained with ROC AUC: {roc_auc:.2f}")
    logging.info(classification_report(y_test, y_pred))

