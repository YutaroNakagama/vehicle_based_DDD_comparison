import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_top_features_by_importance(X: pd.DataFrame, y: pd.Series, top_k: int = 10) -> list:
    """Select top-k features based on RandomForest feature importances."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X, y)
    importances = clf.feature_importances_
    feature_ranking = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    return [name for name, _ in feature_ranking[:top_k]]

