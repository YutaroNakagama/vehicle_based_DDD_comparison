# data/

Raw and processed datasets (ignored by git).

## Directory Structure

```
data/
├── interim/               # 中間データ（前処理パイプライン途中）
│   ├── eeg/              # EEG信号データ
│   │   ├── common/       # 共通前処理済み
│   │   ├── Lstm/         # LSTM用加工
│   │   ├── SvmA/         # SVM (all features) 用
│   │   └── SvmW/         # SVM (wavelet) 用
│   ├── merged/           # EEG + 車両データ マージ済み
│   │   └── (同上)
│   ├── smooth_std_pe/    # スムージング・標準化・PE処理
│   ├── time_freq_domain/ # 時間-周波数領域特徴量
│   └── wavelet/          # ウェーブレット変換係数
└── processed/            # 最終処理済み（モデル入力用）
    ├── common/           # 共通特徴量セット
    ├── Lstm/             # LSTM用シーケンスデータ
    ├── SvmA/             # SVM用特徴量（全特徴）
    └── SvmW/             # SVM用特徴量（ウェーブレット）
```

## File Naming Convention

- `processed_S0XXX_Y.csv` - 被験者ID `S0XXX`、セッション `Y` (1 or 2)
- 例: `processed_S0113_1.csv` → 被験者S0113の第1セッション

## Data Flow

1. **Raw** → `interim/eeg/` (EEG前処理)
2. `interim/eeg/` + 車両データ → `interim/merged/`
3. `interim/merged/` → 特徴量抽出 → `interim/smooth_std_pe/`, `time_freq_domain/`, `wavelet/`
4. 特徴量統合 → `processed/` (モデル別に分岐)

## Notes

- 全データファイルは `.gitignore` により Git 管理対象外
- 実データは JAIST KAGAYAKI クラスタ上で管理
