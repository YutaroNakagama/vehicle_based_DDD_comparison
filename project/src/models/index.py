
import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score


# Calculation of feature indices (e.g., Fisher, as per the previous implementation)
def calculate_feature_indices(X, y):
    indices = {"Fisher_Index": [], "Correlation_Index": [], "T-test_Index": [], "Mutual_Information_Index": []}
    y_classes = [0, 1]
    for i in range(X.shape[1]):
        xi = X.iloc[:, i]
        mu0 = xi[y == y_classes[0]].mean()
        mu1 = xi[y == y_classes[1]].mean()
        sigma0 = xi[y == y_classes[0]].std()
        sigma1 = xi[y == y_classes[1]].std()
        n0 = sum(y == y_classes[0])
        n1 = sum(y == y_classes[1])

        fisher_index = abs(mu1 - mu0) / (sigma1**2 + sigma0**2 + 1e-6)
        indices["Fisher_Index"].append(fisher_index)

        correlation_index = np.cov(xi, y)[0, 1] / (np.std(xi) * np.std(y) + 1e-6)
        indices["Correlation_Index"].append(correlation_index)

        t_test_index = abs(mu1 - mu0) / np.sqrt((sigma1**2 / n1) + (sigma0**2 / n0) + 1e-6)
        indices["T-test_Index"].append(t_test_index)

        mutual_info = mutual_info_score(xi, y)
        indices["Mutual_Information_Index"].append(mutual_info)

    return pd.DataFrame(indices, index=X.columns)
