# src/utils/data/split.py

from sklearn.model_selection import train_test_split

def data_split(df):
    df = df[df["KSS_Theta_Alpha_Beta"].isin([1, 2, 8, 9])]

    feature_columns = df.columns[1:46].tolist()
    if 'subject_id' in df.columns:
        feature_columns.append('subject_id')

    X = df[feature_columns].dropna()
    y = df.loc[X.index, "KSS_Theta_Alpha_Beta"].replace({1: 0, 2: 0, 8: 1, 9: 1})

    # Train/val/test split: 60/20/20
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

