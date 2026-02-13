# 実験3（先行研究再現）の実験条件

このファイルは「実験3：先行研究モデルの再現（split2 ドメイン分割版）」で使用した実験条件の一覧を示します。

(内容は `experiment_3_conditions.md` から移行されています — 旧ファイル名: `experiment_3_conditions.md`)

---

<!-- 以下、元ファイルの内容をそのまま移行しています -->

## 概要

- **目的**: 先行研究のモデル（SvmA, SvmW, Lstm）を split2 ドメイン分割で再現し、ドメインシフト耐性を評価
- **モデル**: SvmW, SvmA, Lstm（3 種）
- **データ分割**: `split2`（`in_domain`: 44 名、`out_domain`: 43 名）
- **ランチャー**: `scripts/hpc/launchers/launch_prior_research_split2.sh`
- **ジョブスクリプト**: `scripts/hpc/jobs/train/pbs_prior_research_split2.sh`

...（元ファイルの詳細はそのまま移行）
