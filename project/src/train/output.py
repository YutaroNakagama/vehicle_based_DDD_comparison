
def show_result(name,result):
    print(f"\nClassifier: {name}")
    print(f"Optimized Parameters: {result['optimized_params']}")
    print(f"Selected Features: {result['selected_features']}")
    print(f"Mean Squared Error: {result['mse']}")
    print(f"Classification Report:\n{result['classification_report']}")
    print(f"ROC AUC: {result['roc_auc']:.2f}")
