from src.config import SUBJECT_LIST_PATH, PROCESS_CSV_PATH, MODEL_PKL_PATH, OUTPUT_SVG_PATH
from src.utils.loaders import read_subject_list
#from src.utils.merge import combine_file
from src.train.index import calculate_feature_indices
from src.train.anfis import calculate_id
from src.evaluation.lstm import lstm_eval
from src.evaluation.SvmA import SvmA_eval
from src.train.output import show_result

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning Libraries
import xgboost as xgb
import lightgbm as lgb
import pickle
from pyswarm import pso
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# Model Selection and Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, mean_squared_error, mutual_info_score

# Data Preprocessing
from sklearn.preprocessing import StandardScaler

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Global variables to track progress
iteration_counter = 0
previous_score = float('inf') # Initially set to a very high value
no_improvement_counter = 0
early_stopping_threshold = 1 # Threshold for the number of iterations without improvement
optimization_results = {}
dfs = []

def objective_function_with_progress(params):
    """
    Add progress tracking to the PSO objective function
    """
    global iteration_counter
    iteration_counter += 1

    # Logic for the objective function
    threshold = params[0] # Use the threshold as one of the optimised parameters
    ids = calculate_id(feature_indices, params[1:])
    selected_feature_indices = np.where(ids > threshold)[0]
    selected_features = X_train.columns[selected_feature_indices]
    if len(selected_features) == 0:
        print(f"Threshold {threshold:.2f} results in no selected features.")
        return 1e6

    X_selected_train = X_train[selected_features]
    X_selected_test = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_selected_train)
    X_test_scaled = scaler.transform(X_selected_test)

    clf.fit(X_train_scaled, y_train_binary)

    # Calculate the training accuracy
    y_train_pred = clf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_binary, y_train_pred)

    # Output progress
    #print(f"Training Accuracy for {name}: {train_accuracy:.4f}")

    y_pred = clf.predict(X_test_scaled)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test_binary, y_pred)

    # Output progress
    print(f"Iteration: {iteration_counter}, Params: {params}, MSE: {mse}")
    return mse

def show_result(name,result):
    print(f"\nClassifier: {name}")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Classification Report:\n{result['classification_report']}")
    print(f"ROC AUC: {result['roc_auc']:.2f}")

def combine_file(subject, model):
    subject_id, version = subject.split('/')[0], subject.split('/')[1].split('_')[-1]
    #subject_id, version = subject
    file_name = f'processed_{subject_id}_{version}.csv'
    try:
        df = pd.read_csv(f'{PROCESS_CSV_PATH}/{model}/{file_name}')
        dfs.append(df)
        print(f"File found: {file_name}")
    except FileNotFoundError:
        print(f"File not found: {file_name}")

def optimizar(name,clf,model):
    print(f"Optimizing for classifier: {name}")

    # Optimise using PSO
    #lb, ub = [0.1] * 6, [1.0] * 6  # 下限と上限
    lb = [0.5] + [0.1] * (len(feature_indices) - 1)  # 閾値の下限を0.5に設定
    ub = [0.9] + [1.0] * (len(feature_indices) - 1)  # 閾値の上限を0.9に設定
    optimized_params, best_score = pso(objective_function_with_progress, lb, ub, swarmsize=20, maxiter=1, debug=False)
    
    # Save results after optimisation
    ids = calculate_id(feature_indices, optimized_params)
    selected_feature_indices = np.where(ids > 0.8)[0]  # 条件に一致するインデックスを取得
    selected_features = X_train.columns[selected_feature_indices]
    
    # Retrain the model with optimised features
    X_selected_train = X_train[selected_features]
    X_selected_test = X_test[selected_features]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_selected_train)
    X_test_scaled = scaler.transform(X_selected_test)

    clf.fit(X_train_scaled, y_train_binary)

    model_filename = f"{MODEL_PKL_PATH}/{model}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(clf, f)

    sys.exit()

    # 訓練データでの予測と精度計算
    y_train_pred = clf.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_binary, y_train_pred) 

    # Display training accuracy
    print(f"Training Accuracy for {name}: {train_accuracy:.4f}")

    y_pred = clf.predict(X_test_scaled)

    try:
        if hasattr(clf, "predict_proba"):
            y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]  # 確率スコア
        elif hasattr(clf, "decision_function"):
            y_pred_proba = clf.decision_function(X_test_scaled)  # 決定関数
        else:
            raise AttributeError(f"{name} does not support probability or decision score output.")
    
        # Calculate ROC curve and AUC score
        fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
        roc_auc = auc(fpr, tpr)

    except AttributeError as e:
        print(f"Skipping ROC and AUC calculation for {name}: {str(e)}")
        roc_auc = 0

    mse = mean_squared_error(y_test_binary, y_pred)
    report = classification_report(y_test_binary, y_pred)
    
    # Save the results
    optimization_results[name] = {
        "optimized_params": optimized_params,
        "selected_features": selected_features.tolist(),
        "mse": mse,
        "classification_report": report,
        "roc_auc": roc_auc
    }

def eval_pipeline(model):
    global X_train, X_test, y_train, y_test, y_train_binary, y_test_binary
    global clf, name
    global feature_indices

    # Read subject list
    #with open(SUBJECT_LIST_PATH, 'r') as file:
    #    subjects = [line.strip().split('/') for line in file.readlines()]

    subject_list = read_subject_list()
    # Create dataframe list and read csv file
    data_model = model if model in {"SvmW", "SvmA", "Lstm"} else "common"
    for subject in subject_list:
        combine_file(subject, data_model)

    # Combine the data and filter for classes 4 and 8
    all_data = pd.concat(dfs, ignore_index=True)
    filtered_data = all_data[all_data["KSS_Theta_Alpha_Beta"].isin([1,2,8,9])]

    # Select features and target, removing missing values
    X = filtered_data.iloc[:, 1:46].dropna()
    y = filtered_data["KSS_Theta_Alpha_Beta"].loc[X.index]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Convert target variable into binary classes (4 -> 0, 8 -> 1)
    y_train_binary = y_train.replace({**dict.fromkeys([1, 2], 0), **dict.fromkeys([8, 9], 1)})
    y_test_binary = y_test.replace({**dict.fromkeys([1, 2], 0), **dict.fromkeys([8, 9], 1)})

    if model == 'Lstm':
        lstm_eval(X_test, y_test_binary)
    else:
        feature_indices = calculate_feature_indices(X_train, y_train_binary)
    
        # Define multiple classifiers
        if model == 'SvmA':
            print("call svmA train")
            SvmA_eval(X_train, X_test, y_train, y_test, feature_indices)
        else:
            if model == 'RF':
                classifiers = {
                    # 1. Tree-based algorithms
                    #"Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(random_state=42),
                    #"AdaBoost": AdaBoostClassifier(random_state=42),
                    #"Gradient Boosting": GradientBoostingClassifier(random_state=42),
                    #"XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
                    #"LightGBM": lgb.LGBMClassifier(random_state=42),
                    #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
                
                    # 2. Linear models
                    #"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    #"Perceptron": Perceptron(max_iter=1000, random_state=42),
                
                    # 3. SVM (Support Vector Machines)
                    #"SVM (Linear Kernel)": SVC(kernel="linear", probability=True, random_state=42),
                    #"SVM (RBF Kernel)": SVC(kernel="rbf", probability=True, random_state=42),
                
                    # 4. k-Nearest Neighbours
                    #"K-Nearest Neighbors": KNeighborsClassifier(),
                
                    # 5. Neural Networks
                    #"MLP (Neural Network)": MLPClassifier(max_iter=500, random_state=42),
                }
            
            
                model_filename = f"{MODEL_PKL_PATH}/{model}.pkl"
                feat_filename = f"{MODEL_PKL_PATH}/{model}_feat.npy"
            
                with open(model_filename, "rb") as f:
                    clf = pickle.load(f)
            
                selected_features = np.load(feat_filename, allow_pickle=True)
            
            
                # Display training accuracy
                X_selected_train = X_train[selected_features]
                X_selected_test = X_test[selected_features]
            
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_selected_train)
                X_test_scaled = scaler.transform(X_selected_test)
            
                y_pred = clf.predict(X_test_scaled)
            
                try:
                    if hasattr(clf, "predict_proba"):
                        y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]  # 確率スコア
                    elif hasattr(clf, "decision_function"):
                        y_pred_proba = clf.decision_function(X_test_scaled)  # 決定関数
                    else:
                        raise AttributeError(f"{name} does not support probability or decision score output.")
                
                    # Calculate ROC curve and AUC score
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
            
                except AttributeError as e:
                    print(f"Skipping ROC and AUC calculation for {name}: {str(e)}")
                    roc_auc = 0
            
                mse = mean_squared_error(y_test_binary, y_pred)
                report = classification_report(y_test_binary, y_pred)
            
                print('mse',mse)
                print('report',report)
                print('roc_auc',roc_auc)
                
