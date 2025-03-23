import numpy as np
import pandas as pd

def generate_domain_labels(subject_list, X):
    return X['subject_id'].values

def domain_mixup(X, y, domain_labels, alpha=0.2, augment_ratio=0.3):

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    domain_labels = pd.Series(domain_labels).reset_index(drop=True).values

    unique_domains = np.unique(domain_labels)
    augmented_X = []
    augmented_y = []

    # 数値カラムだけを抽出
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    num_augment = int(len(X) * augment_ratio)

    for _ in range(num_augment):
        dom1, dom2 = np.random.choice(unique_domains, 2, replace=False)

        dom1_indices = np.where(domain_labels == dom1)[0]
        dom2_indices = np.where(domain_labels == dom2)[0]
    
        if len(dom1_indices) == 0 or len(dom2_indices) == 0:
            continue        

        idx1 = np.random.choice(np.where(domain_labels == dom1)[0])
        idx2 = np.random.choice(np.where(domain_labels == dom2)[0])

        lam = np.random.beta(alpha, alpha)

        # 数値カラムだけをMixup
        new_numeric = lam * X.iloc[idx1][numeric_columns].values + \
                      (1 - lam) * X.iloc[idx2][numeric_columns].values

        # ラベルはランダムに片方から選ぶ
        new_y = y.iloc[idx1] if np.random.rand() < lam else y.iloc[idx2]

        # 一時的なDataFrameで保存
        new_X = pd.DataFrame([new_numeric], columns=numeric_columns)
        augmented_X.append(new_X)
        augmented_y.append(new_y)

    # concatで整形
    X_aug_numeric = pd.concat(augmented_X, ignore_index=True)
    y_aug = pd.Series(augmented_y)

    # 数値カラム以外（例：subject_id）をXから取り出してそのまま連結
    non_numeric = X.drop(columns=numeric_columns).reset_index(drop=True)
    X_numeric = X[numeric_columns].reset_index(drop=True)

    X_combined = pd.concat([X_numeric, X_aug_numeric], ignore_index=True)
    y_combined = pd.concat([y.reset_index(drop=True), y_aug], ignore_index=True)

    # 数値以外の列（例：subject_id）も戻す（元の行を繰り返す or NaNにしてもOK）
    if not non_numeric.empty:
        non_numeric_aug = non_numeric.sample(n=len(X_aug_numeric), replace=True).reset_index(drop=True)
        non_numeric_combined = pd.concat([non_numeric, non_numeric_aug], ignore_index=True)
        X_combined = pd.concat([X_combined, non_numeric_combined], axis=1)

    return X_combined, y_combined

