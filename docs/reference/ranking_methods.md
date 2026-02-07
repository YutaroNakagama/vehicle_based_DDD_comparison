# Subject Ranking Methods for Domain Analysis

## Overview

This document describes the **subject ranking methods** implemented for domain analysis in driver drowsiness detection. These methods rank subjects based on their domain similarity/dissimilarity to identify:

- **out_domain**: Subjects with atypical patterns (outliers, far from center)
- **mid_domain**: Subjects with intermediate patterns
- **in_domain**: Subjects with typical patterns (near center, similar to majority)

This ranking is crucial for domain generalization experiments, where we evaluate how models trained on typical subjects generalize to atypical ones.

---

## Implemented Ranking Methods

### 1. Mean Distance (`mean_distance`)

**Description**: Baseline method that calculates the average distance from each subject to all other subjects.

**Formula**:
$$d_i^{mean} = \frac{1}{n-1} \sum_{j \neq i} D_{ij}$$

where $D_{ij}$ is the distance between subjects $i$ and $j$.

**Characteristics**:
- Simple and interpretable
- Sensitive to all pairwise distances
- May be affected by outliers in the distance matrix

**References**:
- Ben-David, S., et al. "A theory of learning from different domains." *Machine Learning*, 79(1-2), 151-175, 2010. [[link]](https://link.springer.com/article/10.1007/s10994-009-5152-4)

---

### 2. Centroid Distance in UMAP Space (`centroid_umap`)

**Description**: Projects subjects into 2D UMAP space and measures Euclidean distance from each subject to the global centroid.

**Algorithm**:
1. Apply UMAP dimensionality reduction to the distance matrix
2. Compute centroid of all projected points: $\mathbf{c} = \frac{1}{n}\sum_i \mathbf{x}_i$
3. Rank subjects by Euclidean distance to centroid: $d_i = ||\mathbf{x}_i - \mathbf{c}||_2$

**Characteristics**:
- Captures non-linear manifold structure
- Provides intuitive geometric interpretation
- Visualization-friendly

**References**:
- McInnes, L., Healy, J., & Melville, J. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction." *arXiv preprint arXiv:1802.03426*, 2018. [[link]](https://arxiv.org/abs/1802.03426)

---

### 3. Local Outlier Factor (`lof`)

**Description**: Density-based outlier detection method that compares the local density of a point to the local densities of its neighbors.

**Formula**:
$$LOF_k(A) = \frac{\sum_{B \in N_k(A)} \frac{lrd_k(B)}{lrd_k(A)}}{|N_k(A)|}$$

where $lrd_k(A)$ is the local reachability density of point $A$ and $N_k(A)$ is its k-nearest neighbors.

**Characteristics**:
- Identifies local outliers that may not be global outliers
- Effective for non-uniform density distributions
- Parameter $k$ controls neighborhood size

**Implementation Details**:
- Uses `sklearn.neighbors.LocalOutlierFactor`
- Default: `n_neighbors=20`, `contamination='auto'`
- Distance matrix converted to feature space via MDS

**References**:
- Breunig, M. M., et al. "LOF: Identifying Density-Based Local Outliers." *ACM SIGMOD Record*, 29(2), 93-104, 2000. [[link]](https://dl.acm.org/doi/10.1145/335191.335388)

---

### 4. K-Nearest Neighbors Average Distance (`knn`)

**Description**: Calculates the average distance to the k nearest neighbors for each subject.

**Formula**:
$$d_i^{knn} = \frac{1}{k} \sum_{j \in N_k(i)} D_{ij}$$

where $N_k(i)$ is the set of k nearest neighbors of subject $i$.

**Characteristics**:
- More robust than mean distance (ignores distant points)
- Captures local neighborhood structure
- Parameter $k$ controls local vs global sensitivity

**Implementation Details**:
- Default: $k = \min(10, n-1)$
- Higher $d_i^{knn}$ indicates subjects in sparse regions (potential outliers)

**References**:
- Cover, T., & Hart, P. "Nearest neighbor pattern classification." *IEEE Transactions on Information Theory*, 13(1), 21-27, 1967. [[link]](https://ieeexplore.ieee.org/document/1053964)
- Ramaswamy, S., Rastogi, R., & Shim, K. "Efficient algorithms for mining outliers from large data sets." *ACM SIGMOD Record*, 29(2), 427-438, 2000. [[link]](https://dl.acm.org/doi/10.1145/335191.335437)

---

### 5. Median Distance (`median_distance`)

**Description**: Uses the median instead of mean distance, providing robustness to extreme values.

**Formula**:
$$d_i^{median} = \text{median}_{j \neq i}(D_{ij})$$

**Characteristics**:
- Robust to outliers in the distance matrix
- Less sensitive to extreme pairwise distances
- Good for heavy-tailed distance distributions

**References**:
- Rousseeuw, P. J., & Croux, C. "Alternatives to the Median Absolute Deviation." *Journal of the American Statistical Association*, 88(424), 1273-1283, 1993. [[link]](https://www.tandfonline.com/doi/abs/10.1080/01621459.1993.10476408)

---

### 6. Isolation Forest (`isolation_forest`)

**Description**: Tree-based anomaly detection method that isolates outliers by random partitioning.

**Algorithm**:
1. Build random trees by selecting random features and random split values
2. Anomaly score = average path length to isolate each point
3. Outliers require fewer splits (shorter paths) to isolate

**Formula**:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

where $h(x)$ is the path length and $c(n)$ is the average path length of unsuccessful search in a binary search tree.

**Characteristics**:
- Efficient for high-dimensional data
- Does not assume any particular distribution
- Linear time complexity: $O(t \cdot n \log n)$ where $t$ is number of trees

**Implementation Details**:
- Uses `sklearn.ensemble.IsolationForest`
- Default: `n_estimators=100`, `contamination='auto'`
- Distance matrix converted to feature space via MDS

**References**:
- Liu, F. T., Ting, K. M., & Zhou, Z. H. "Isolation Forest." *IEEE International Conference on Data Mining (ICDM)*, 413-422, 2008. [[link]](https://ieeexplore.ieee.org/document/4781136)
- Liu, F. T., Ting, K. M., & Zhou, Z. H. "Isolation-Based Anomaly Detection." *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(1), 1-39, 2012. [[link]](https://dl.acm.org/doi/10.1145/2133360.2133363)

---

## Method Comparison

| Method | Type | Complexity | Strengths | Weaknesses |
|--------|------|------------|-----------|------------|
| `mean_distance` | Global | O(n²) | Simple, interpretable | Sensitive to outliers |
| `centroid_umap` | Manifold | O(n²) | Non-linear structure | Projection artifacts |
| `lof` | Density-based | O(n² log n) | Local outlier detection | Sensitive to k |
| `knn` | Local | O(n²) | Robust, local structure | Choice of k |
| `median_distance` | Global | O(n² log n) | Robust to outliers | Less discriminative |
| `isolation_forest` | Tree-based | O(n log n) | Efficient, scalable | Stochastic results |

---

## Evaluation Results

Based on our experiments with driver drowsiness detection data:

| Ranking Method | AUPRC | F2 | Recall | Best Mode |
|----------------|-------|-----|--------|-----------|
| KNN | **0.1403** | **0.217** | **0.54** | target_only |
| Isolation Forest | 0.1185 | 0.169 | 0.42 | target_only |
| LOF | 0.1138 | 0.170 | 0.42 | target_only |
| Centroid UMAP | 0.1100 | 0.163 | 0.40 | target_only |
| Median Distance | 0.0890 | 0.125 | 0.28 | target_only |
| Mean Distance | 0.0796 | 0.110 | 0.24 | target_only |

**Key Findings**:
- **KNN** method performs best across all metrics
- **target_only** training mode consistently outperforms pooled and source_only
- Density-based methods (LOF, KNN) outperform global distance methods

---

## Usage

### Generate Rankings

```bash
# Generate rankings for all 6 methods
python scripts/python/domain_analysis/generate_new_rankings.py \
    --distance mmd \
    --outdir results/domain_analysis/ranks29

# Generate rankings with visualization
python scripts/python/domain_analysis/generate_new_rankings.py \
    --distance mmd \
    --outdir results/domain_analysis/ranks29 \
    --plot
```

### Output Files

Rankings are saved in:
```
results/domain_analysis/ranks29/{method}/
├── {distance}_out_domain.txt    # Outlier subjects (29)
├── {distance}_mid_domain.txt    # Intermediate subjects (29)
├── {distance}_in_domain.txt     # Typical subjects (29)
└── ranks29_names.txt            # Method metadata
```

---

## References (General)

### Domain Adaptation & Generalization
1. Ben-David, S., et al. "A theory of learning from different domains." *Machine Learning*, 79(1-2), 151-175, 2010.
2. Ganin, Y., et al. "Domain-Adversarial Training of Neural Networks." *JMLR*, 17(1), 2096-2030, 2016.
3. Zhou, K., et al. "Domain Generalization: A Survey." *IEEE TPAMI*, 2022.

### Outlier Detection
4. Chandola, V., Banerjee, A., & Kumar, V. "Anomaly Detection: A Survey." *ACM Computing Surveys*, 41(3), 1-58, 2009.
5. Aggarwal, C. C. "Outlier Analysis." *Springer*, 2017.

### Driver Drowsiness Detection
6. Sahayadhas, A., Sundaraj, K., & Murugappan, M. "Detecting driver drowsiness based on sensors: A review." *Sensors*, 12(12), 16937-16953, 2012.
7. Sikander, G., & Anwar, S. "Driver fatigue detection systems: A review." *IEEE TITS*, 20(6), 2339-2352, 2019.

---

## Implementation

The ranking methods are implemented in:
- `src/analysis/clustering_projection_ranked.py` — Core ranking algorithms
- `scripts/python/domain_analysis/generate_new_rankings.py` — CLI interface

Each method follows a consistent interface:
```python
def rank_by_{method}(matrix: np.ndarray, subjects: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Rank subjects using {method}.
    
    Returns:
        (out_domain_subjects, mid_domain_subjects, in_domain_subjects)
    """
```
