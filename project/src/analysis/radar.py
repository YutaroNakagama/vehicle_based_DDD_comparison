from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # HPCでも安全
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def make_radar(
    wide_df: pd.DataFrame,
    out_dir: Path | str,
    metrics: list[str],
    ylim: tuple[float, float] = (0.0, 1.0),
) -> Path:
    """
    wide_df には 'group' と '{metric}_finetune' / '{metric}_only10' 列が必要。
    各グループのpngと、まとめPDFを out_dir に出力。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "radar_finetune_vs_only10.pdf"

    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    pdf = PdfPages(pdf_path)
    saved = 0
    for _, row in wide_df.iterrows():
        group = str(row["group"])
        fins = [row.get(f"{m}_finetune", np.nan) for m in metrics]
        onls = [row.get(f"{m}_only10",   np.nan) for m in metrics]
        if all(np.isnan(fins)) and all(np.isnan(onls)):
            continue
        fins = fins + fins[:1]
        onls = onls + onls[:1]

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics)
        ax.set_ylim(*ylim)
        ax.plot(angles, fins, linewidth=2, label="finetune"); ax.fill(angles, fins, alpha=0.1)
        ax.plot(angles, onls, linewidth=2, linestyle="--", label="only10"); ax.fill(angles, onls, alpha=0.1)
        ax.set_title(f"Group: {group}", va="bottom")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

        png_path = out_dir / f"radar_{group}.png"
        fig.savefig(png_path, bbox_inches="tight", dpi=150)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    pdf.close()
    print(f"[radar] Saved {saved} images to {out_dir}")
    print(f"[radar] Combined PDF: {pdf_path}")
    return pdf_path
