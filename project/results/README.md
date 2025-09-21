# 📄 `project/results/README.md`

```markdown
# results/ directory

This directory stores **experiment outputs** including metrics, distance matrices, rankings, and analysis results.  
Each subdirectory corresponds to a specific type of result.

---

## Structure
```

results/
├── analysis/        # Aggregated analysis outputs (tables, plots, correlations)
├── distances/       # Computed distance matrices (NumPy .npy files, JSON metadata)
├── dtw/             # Dynamic Time Warping (DTW) distance results
├── group\_distances/ # Distances grouped by subject clusters
├── mmd/             # Maximum Mean Discrepancy (MMD) results
├── ranks/           # Ranking results (CSV)
├── ranks10/         # Ranking results (10-subject experiments)
├── ranks20/         # Ranking results (20-subject experiments)
└── wasserstein/     # Wasserstein distance results

```

---

## Notes
- **analysis/** contains aggregated CSVs and figures (e.g., radar charts, summary tables).  
- **distances/** stores core distance matrices (`*_matrix.npy`) and subject mapping files (`subjects.json`).  
- **dtw/**, **mmd/**, **wasserstein/** are separated for clarity by metric type.  
- **ranks/**, **ranks10/**, **ranks20/** hold ranking-based evaluations.  

---

## Tips
- Always record metadata alongside results (e.g., config, seed, sample size) to ensure reproducibility.  
- Use `project/misc/aggregate_*.py` scripts to consolidate multiple result files.  
