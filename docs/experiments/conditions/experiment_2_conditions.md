# 実験2（ドメインシフト）の実験条件

このファイルは「実験2：ドメインシフト (split2)」で使用した実験条件の一覧を示します。

## 概要
- モデル: BalancedRF (Random Forest のバランスサンプル設定)
- データ分割: `split2`（2グループ：`in_domain`（44名）、`out_domain`（43名））

## パラメータ（組合せ）

- 距離指標 (DISTANCE): `mmd`, `wasserstein`, `dtw` (3種)
- 実験モード (MODE): `source_only`, `target_only` (2種)
- ドメイン指示 (DOMAIN): `in_domain`, `out_domain` (2種)
- 乱数シード (SEED): 16個（例: 0..15）

計算: 3 (DISTANCE) × 2 (MODE) × 2 (DOMAIN) × 16 (SEED) = 192 ジョブ

この設計により、split2 の全ケースを網羅してドメイン間の一般化性能を評価します。

## ジョブ例（PBS 環境）

例：`pbs_domain_comparison_split2.sh` を使った qsub 呼び出し

```bash
qsub -v CONDITION=baseline,MODE=source_only,DISTANCE=mmd,DOMAIN=out_domain,SEED=0 \
    scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh
```

上のコマンドをループで回すことで 192 ジョブを投入します（launcher スクリプト参照）。

## ファイルと出力先

- ランキング入力（split2）: `results/analysis/domain/distance/subject-wise/ranks/split2/{ranking_method}/`
- ジョブスクリプト: `scripts/hpc/jobs/domain_analysis/pbs_domain_comparison_split2.sh`
- 実験ログ・評価: `results/outputs/evaluation/BalancedRF/`（モデルごとのジョブ結果）

## 備考
- 実行前に `results/analysis/domain/` にランキングファイル（`*_in_domain.txt` / `*_out_domain.txt`）が生成されていることを確認してください。
- 実行用のランチャー（bulk submit）やリソース（ノード数、時間）は `scripts/hpc/launchers/` の設定に依存します。

---

作成日: 2026-02-07
