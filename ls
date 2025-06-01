[1mdiff --git a/project/src/config.py b/project/src/config.py[m
[1mindex e64e0eb..2957028 100644[m
[1m--- a/project/src/config.py[m
[1m+++ b/project/src/config.py[m
[36m@@ -22,7 +22,7 @@[m [mOUTPUT_SVG_PATH         = './output/svg'[m
 # Choices for processing and modeling[m
 DATA_PROCESS_CHOICES = ["SvmA", "SvmW", "Lstm", "common"][m
 MODEL_CHOICES = [[m
[31m-    "SvmA", "SvmW", "Lstm", "RF", "DecisionTree", "AdaBoost", "GradientBoosting",[m
[32m+[m[32m    "SvmA", "SvmW", "Lstm", "RF", "BalancedRF", "DecisionTree", "AdaBoost", "GradientBoosting",[m
     "XGBoost", "LightGBM", "CatBoost", "LogisticRegression", "SVM", "K-Nearest Neighbors", "MLP"[m
 ][m
 [m
[36m@@ -48,4 +48,4 @@[m [mWAVELET_LEV = 3[m
 TOP_K_FEATURES = 10[m
 [m
 # Optuna [m
[31m-N_TRIALS = 15[m
[32m+[m[32mN_TRIALS = 30[m
[1mdiff --git a/project/src/models/architectures/common.py b/project/src/models/architectures/common.py[m
[1mindex c9e72cd..e311981 100644[m
[1m--- a/project/src/models/architectures/common.py[m
[1m+++ b/project/src/models/architectures/common.py[m
[36m@@ -25,6 +25,7 @@[m [mfrom sklearn.linear_model import LogisticRegression[m
 from sklearn.svm import SVC[m
 from sklearn.neighbors import KNeighborsClassifier[m
 from sklearn.neural_network import MLPClassifier[m
[32m+[m[32mfrom imblearn.ensemble import BalancedRandomForestClassifier[m
 [m
 from src.models.feature_selection.anfis import calculate_id[m
 from src.config import MODEL_PKL_PATH, N_TRIALS[m
[36m@@ -99,10 +100,26 @@[m [mdef common_train([m
                 "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),[m
                 "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),[m
                 "random_state": 42,[m
[31m-                "class_weight": "balanced"[m
[32m+[m[32m                "class_weight": "balanced_subsample"[m
             }[m
             clf = RandomForestClassifier(**params)[m
 [m
[32m+[m[32m        elif model == "BalancedRF":[m
[32m+[m[32m            sampling_strategy = trial.suggest_categorical([m
[32m+[m[32m                "sampling_strategy", ["auto", "majority", "not majority", "not minority", "all", 0.5, 0.75, 1.0][m
[32m+[m[32m            )[m
[32m+[m[32m            params = {[m
[32m+[m[32m                "n_estimators": trial.suggest_int("n_estimators", 100, 300),[m
[32m+[m[32m                "max_depth": trial.suggest_int("max_depth", 5, 30),[m
[32m+[m[32m                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),[m
[32m+[m[32m                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),[m
[32m+[m[32m                "sampling_strategy": sampling_strategy,[m[41m  [m
[32m+[m[32m                "replacement": trial.suggest_categorical("replacement", [True, False]),[m
[32m+[m[32m                "random_state": 42,[m
[32m+[m[32m                # NOTE: class_weight is not needed; B-RF handles balancing internally[m
[32m+[m[32m            }[m
[32m+[m[32m            clf = BalancedRandomForestClassifier(**params)[m
[32m+[m
         elif model == "CatBoost":[m
             params = {[m
                 "iterations": trial.suggest_int("iterations", 100, 300),[m
[36m@@ -218,7 +235,9 @@[m [mdef common_train([m
     elif model == "CatBoost":[m
         best_clf = CatBoostClassifier(**best_params)[m
     elif model == "RF":[m
[31m-        best_clf = RandomForestClassifier(**best_params, class_weight="balanced")[m
[32m+[m[32m        best_clf = RandomForestClassifier(**best_params, class_weight="balanced_subsample")[m
[32m+[m[32m    elif model == "BalancedRF":[m
[32m+[m[32m        best_clf = BalancedRandomForestClassifier(**best_params)[m
     elif model == "LogisticRegression":[m
         best_clf = LogisticRegression(**best_params)[m
     elif model == "SVM":[m
[36m@@ -275,3 +294,30 @@[m [mdef common_train([m
     logging.info(f"{model} (Optuna) trained with ROC AUC: {roc_auc:.2f}")[m
     logging.info(classification_report(y_test, y_pred))[m
 [m
[32m+[m[32m    # Threshold optimization (F1-score)[m
[32m+[m[32m    if y_pred_proba is not None:[m
[32m+[m[32m        from sklearn.metrics import precision_recall_curve, f1_score[m
[32m+[m
[32m+[m[32m        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)[m
[32m+[m[32m        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)[m
[32m+[m[32m        best_idx = np.argmax(f1_scores)[m
[32m+[m[32m        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5[m
[32m+[m
[32m+[m[32m        logging.info(f"Optimal threshold for F1: {best_threshold:.3f}")[m
[32m+[m[41m        [m
[32m+[m[32m        y_pred_opt = (y_pred_proba >= best_threshold).astype(int)[m
[32m+[m[32m        logging.info("Classification Report with optimized threshold:")[m
[32m+[m[32m        logging.info(classification_report(y_test, y_pred_opt))[m
[32m+[m
[32m+[m[32m        # Save threshold[m
[32m+[m[32m        threshold_meta = {[m
[32m+[m[32m            "model": model,[m
[32m+[m[32m            "threshold": best_threshold,[m
[32m+[m[32m            "metric": "F1-optimal",[m
[32m+[m[32m        }[m
[32m+[m[32m        with open(f"{MODEL_PKL_PATH}/{model_type}/threshold_{model}{suffix}.json", "w") as f:[m
[32m+[m[32m            json.dump(threshold_meta, f, indent=2)[m
[32m+[m
[32m+[m[32m    else:[m
[32m+[m[32m        logging.warning("Threshold optimization skipped: model does not support probability estimation.")[m
[32m+[m
[1mdiff --git a/project/src/models/architectures/helpers.py b/project/src/models/architectures/helpers.py[m
[1mindex 45a22e3..4ac194e 100644[m
[1m--- a/project/src/models/architectures/helpers.py[m
[1m+++ b/project/src/models/architectures/helpers.py[m
[36m@@ -17,6 +17,7 @@[m [mfrom sklearn.tree import DecisionTreeClassifier[m
 from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier[m
 from sklearn.neighbors import KNeighborsClassifier[m
 from sklearn.neural_network import MLPClassifier[m
[32m+[m[32mfrom imblearn.ensemble import BalancedRandomForestClassifier[m[41m  [m
 [m
 [m
 def get_classifier(model_name: str):[m
[36m@@ -40,6 +41,7 @@[m [mdef get_classifier(model_name: str):[m
     """[m
     classifiers = {[m
         "RF": RandomForestClassifier(random_state=42),[m
[32m+[m[32m        "BalancedRF": BalancedRandomForestClassifier(random_state=42),[m[41m  [m
         "SvmW": SVC(kernel="rbf", probability=True, random_state=42),[m
         "DecisionTree": DecisionTreeClassifier(random_state=42),[m
         "AdaBoost": AdaBoostClassifier(random_state=42),[m
[1mdiff --git a/project/src/models/model_pipeline.py b/project/src/models/model_pipeline.py[m
[1mindex 9eb9dfd..06f2259 100644[m
[1m--- a/project/src/models/model_pipeline.py[m
[1m+++ b/project/src/models/model_pipeline.py[m
[36m@@ -82,8 +82,8 @@[m [mdef train_pipeline([m
     else:[m
         X_train, X_val, X_test, y_train, y_val, y_test = data_split(data)[m
         logging.info(f"X_train shape after data split: {X_train.shape}")[m
[31m-        logging.info(f"X_val shape after data split: {X_train.shape}")[m
[31m-        logging.info(f"X_test shape after data split: {X_test.shape}")[m
[32m+[m[32m        logging.info(f"X_val   shape after data split: {X_val.shape}")[m
[32m+[m[32m        logging.info(f"X_test  shape after data split: {X_test.shape}")[m
 [m
     # Check for label diversity in training set[m
     if y_train.nunique() < 2:[m
[1mdiff --git a/project/src/utils/io/split.py b/project/src/utils/io/split.py[m
[1mindex e78e7ac..ad1c607 100644[m
[1m--- a/project/src/utils/io/split.py[m
[1m+++ b/project/src/utils/io/split.py[m
[36m@@ -31,7 +31,7 @@[m [mdef data_split(df: pd.DataFrame):[m
             - y_val (pd.Series)[m
             - y_test (pd.Series)[m
     """[m
[31m-    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 6, 7, 8, 9])][m
[32m+[m[32m    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 3, 8, 9])][m
 [m
     start_col = "Steering_Range"[m
     end_col = "LaneOffset_AAA"[m
[36m@@ -41,7 +41,7 @@[m [mdef data_split(df: pd.DataFrame):[m
         feature_columns.append('subject_id')[m
 [m
     X = df[feature_columns].dropna()[m
[31m-    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 6: 1, 7: 1, 8: 1, 9: 1})[m
[32m+[m[32m    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 3: 0, 8: 1, 9: 1})[m
 [m
     # Train/val/test split: 60% / 20% / 20%[m
     X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)[m
