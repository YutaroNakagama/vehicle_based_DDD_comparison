import numpy as np

# Gaussian membership function
def gaussian_membership(x, c, s):
    return np.exp(-0.5 * ((x - c) / s) ** 2)

# Calculate ID using ANFIS
def calculate_id(indices, params):
    c, s = params[:len(params)//2], params[len(params)//2:]
    ids = []
    for i, col in enumerate(indices.columns):
        membership = [gaussian_membership(indices[col], c[j], s[j]) for j in range(3)]  # L, M, H
        ids.append(np.max(membership, axis=0)) 
    return np.array(ids)
