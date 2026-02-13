# 実験2（ドメインシフト）の実験条件

このファイルは「実験2：ドメインシフト (split2)」で使用した実験条件の一覧を示します。

(内容は `experiment_2_conditions.md` から移行されています — 旧ファイル名: `experiment_2_conditions.md`)

---

<!-- 以下、元ファイルの内容をそのまま保持 -->

## 概要

- **目的**: split2 ドメイン分割による RF モデルのドメインシフト耐性評価
- **モデル**: RF（BalancedRF は不均衡対策手法として含む）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_paper_domain_split2.sh`（cross/single）、`scripts/hpc/launchers/launch_exp2_mixed.sh`（mixed）
- **ジョブスクリプト**: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`

...（省略せず元のセクションはそのまま移行しています）
