# Subject Ranking Methods for Domain Analysis

## Overview

This document describes the **subject ranking methods** implemented for domain analysis in
driver drowsiness detection. Each method scores every subject-session from a subject
distance matrix (MMD, DTW or W1) and orders them from typical to atypical:

- **out_domain**: Subjects with atypical patterns (outliers, far from the others)
- **in_domain**: Subjects with typical patterns (similar to the majority)
- **mid_domain**: An intermediate group, produced by the three-group variant only

**The paper uses the `knn` method (k = 5) with the two-group split `split2`:** the 43
subject-sessions with the highest scores form the Out group and the other 44 the In group
(see [Grouping used in the paper](#grouping-used-in-the-paper)). The other methods are
implemented for comparison and were not used in the paper.

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
- Default: $k = 5$ (`_rank_by_knn`), the value used in the paper
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

## Grouping used in the paper

The In/Out lists used by the runs are
`results/analysis/exp2_domain_shift/distance/rankings/split2/knn/{mmd,dtw,wasserstein}_{in,out}_domain.txt`,
distributed in the Zenodo bundle together with the distance matrices they were derived
from (`distance/matrices/`). Scoring each subject-session by the mean of its five
smallest distances to the others and taking the 43 highest scores as Out reproduces the
lists exactly (`DATADICTIONARY.md`, Section 5).

## Usage

```bash
# Distance matrices (MMD, Wasserstein, DTW) from the feature tables
python scripts/python/analysis/analyze.py comp-dist     --subject_list config/subjects/subject_list.txt     --data_root data/processed/common     --groups_file config/subjects/target_groups.txt

# Three-group rankings with the methods of this document
python -m src.analysis.clustering_projection_ranked --metric all --ranking knn

# Two-group (split2) kNN lists from globally standardised features
python scripts/python/analysis/domain/regenerate_split2_normalized.py --dry-run
```

`regenerate_split2_normalized.py` recomputes the matrices from the current feature
tables; its lists differ from the ones used by the runs in two or three subject-sessions
per metric (`DATADICTIONARY.md`, Section 5). Run it with `--dry-run` to compare without
overwriting the distributed lists.

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

The ranking methods are implemented in
`src/analysis/clustering_projection_ranked.py` (`_rank_by_{method}` functions and the
command-line entry point `main`).

The kNN ranking, for example:
```python
def _rank_by_knn(matrix: np.ndarray, group_size: int = GROUP_SIZE, k: int = 5
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the rank label of each subject ("out_domain", "mid_domain",
    "in_domain" or "Other") and its mean distance to its k nearest neighbours."""
```
