#!/usr/bin/env python3
"""
Drowsy Timeline Visualization

This script visualizes the temporal distribution of drowsy (KSS >= 8) vs alert 
(KSS <= 5) states for each subject across their experimental session.

The visualization helps understand:
1. When drowsiness occurs during each subject's session
2. Whether drowsiness tends to occur early, middle, or late in sessions
3. The overall class imbalance per subject

Output: results/analysis/imbalance/drowsy_timeline/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from typing import List, Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "common"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "imbalance" / "drowsy_timeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# KSS to binary label mapping (from config)
KSS_LABEL_MAP = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 8: 1, 9: 1}


def load_subject_data(file_path: Path) -> Tuple[str, pd.DataFrame]:
    """Load subject data and extract subject ID."""
    df = pd.read_csv(file_path)
    subject_id = file_path.stem.replace("processed_", "")
    return subject_id, df


def compute_drowsy_label(df: pd.DataFrame) -> np.ndarray:
    """Compute binary drowsy label from KSS score."""
    kss_col = 'KSS_Theta_Alpha_Beta'
    if kss_col not in df.columns:
        logging.warning(f"KSS column not found, trying alternative")
        kss_col = 'KSS_Theta_Alpha_Beta_percent'
    
    if kss_col not in df.columns:
        return np.zeros(len(df))
    
    kss = df[kss_col].values
    # Map KSS to binary: 8-9 = drowsy (1), 1-5 = alert (0), 6-7 = intermediate (exclude or mark as 0)
    labels = np.zeros(len(kss))
    labels[kss >= 8] = 1
    return labels


def analyze_all_subjects() -> Dict[str, Dict]:
    """Analyze all subjects and return summary statistics."""
    csv_files = sorted(DATA_DIR.glob("processed_*.csv"))
    
    results = {}
    for file_path in csv_files:
        subject_id, df = load_subject_data(file_path)
        labels = compute_drowsy_label(df)
        timestamps = df['Timestamp'].values if 'Timestamp' in df.columns else np.arange(len(df))
        
        # Normalize time to percentage of session
        if len(timestamps) > 1:
            time_pct = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min()) * 100
        else:
            time_pct = np.array([50])
        
        results[subject_id] = {
            'labels': labels,
            'timestamps': timestamps,
            'time_pct': time_pct,
            'n_samples': len(labels),
            'n_drowsy': int(labels.sum()),
            'n_alert': int((labels == 0).sum()),
            'drowsy_ratio': labels.mean() if len(labels) > 0 else 0,
            'session_duration_sec': timestamps.max() - timestamps.min() if len(timestamps) > 1 else 0,
        }
        
        # Find drowsy episode timing (start/end percentages)
        drowsy_indices = np.where(labels == 1)[0]
        if len(drowsy_indices) > 0:
            results[subject_id]['first_drowsy_pct'] = time_pct[drowsy_indices[0]]
            results[subject_id]['last_drowsy_pct'] = time_pct[drowsy_indices[-1]]
            results[subject_id]['drowsy_time_pcts'] = time_pct[drowsy_indices]
        else:
            results[subject_id]['first_drowsy_pct'] = None
            results[subject_id]['last_drowsy_pct'] = None
            results[subject_id]['drowsy_time_pcts'] = np.array([])
    
    return results


def plot_individual_timelines(results: Dict[str, Dict], max_subjects: int = 30) -> None:
    """Plot individual subject timelines showing drowsy/alert states."""
    subjects = list(results.keys())[:max_subjects]
    
    fig, axes = plt.subplots(len(subjects), 1, figsize=(14, len(subjects) * 0.5 + 2), 
                             sharex=True)
    if len(subjects) == 1:
        axes = [axes]
    
    for idx, subject_id in enumerate(subjects):
        data = results[subject_id]
        labels = data['labels']
        time_pct = data['time_pct']
        
        ax = axes[idx]
        
        # Create colored segments
        for i in range(len(labels) - 1):
            color = '#e74c3c' if labels[i] == 1 else '#3498db'  # Red for drowsy, blue for alert
            ax.axvspan(time_pct[i], time_pct[i+1], color=color, alpha=0.8)
        
        # Add subject label
        drowsy_pct = data['drowsy_ratio'] * 100
        ax.set_ylabel(f"{subject_id}\n({drowsy_pct:.1f}%)", fontsize=8, rotation=0, 
                     ha='right', va='center')
        ax.set_yticks([])
        ax.set_xlim(0, 100)
    
    axes[-1].set_xlabel('Session Progress (%)', fontsize=12)
    
    # Add legend
    alert_patch = mpatches.Patch(color='#3498db', label='Alert (KSS 1-5)')
    drowsy_patch = mpatches.Patch(color='#e74c3c', label='Drowsy (KSS 8-9)')
    fig.legend(handles=[alert_patch, drowsy_patch], loc='upper right', fontsize=10)
    
    plt.suptitle('Drowsy/Alert State Timeline per Subject', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    output_path = OUTPUT_DIR / "individual_timelines.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_aggregated_heatmap(results: Dict[str, Dict]) -> None:
    """Create a heatmap showing drowsy occurrence across all subjects."""
    # Sort subjects by drowsy ratio
    subjects_sorted = sorted(results.keys(), 
                            key=lambda x: results[x]['drowsy_ratio'], 
                            reverse=True)
    
    # Create time bins (10% intervals)
    n_bins = 20
    bin_edges = np.linspace(0, 100, n_bins + 1)
    
    # Create matrix: subjects x time bins
    matrix = np.zeros((len(subjects_sorted), n_bins))
    
    for i, subject_id in enumerate(subjects_sorted):
        data = results[subject_id]
        labels = data['labels']
        time_pct = data['time_pct']
        
        for j in range(n_bins):
            bin_mask = (time_pct >= bin_edges[j]) & (time_pct < bin_edges[j + 1])
            if bin_mask.sum() > 0:
                matrix[i, j] = labels[bin_mask].mean()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(subjects_sorted) * 0.15)))
    
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels([f"{int(bin_edges[i])}-{int(bin_edges[i+1])}%" 
                        for i in range(n_bins)], rotation=45, ha='right', fontsize=8)
    
    # Show every 5th subject on y-axis for readability
    ytick_step = max(1, len(subjects_sorted) // 20)
    ax.set_yticks(np.arange(0, len(subjects_sorted), ytick_step))
    ax.set_yticklabels([subjects_sorted[i] for i in range(0, len(subjects_sorted), ytick_step)], 
                       fontsize=8)
    
    ax.set_xlabel('Session Progress (%)', fontsize=12)
    ax.set_ylabel('Subject (sorted by drowsy ratio)', fontsize=12)
    ax.set_title('Drowsy Occurrence Heatmap\n(Red = High Drowsy Rate, Blue = Low)', 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Drowsy Proportion', fontsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "drowsy_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_temporal_distribution(results: Dict[str, Dict]) -> None:
    """Plot aggregate distribution of when drowsiness occurs."""
    # Collect all drowsy time percentages
    all_drowsy_pcts = []
    all_alert_pcts = []
    
    for subject_id, data in results.items():
        labels = data['labels']
        time_pct = data['time_pct']
        
        drowsy_mask = labels == 1
        alert_mask = labels == 0
        
        all_drowsy_pcts.extend(time_pct[drowsy_mask])
        all_alert_pcts.extend(time_pct[alert_mask])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of drowsy occurrence timing
    ax1 = axes[0, 0]
    ax1.hist(all_drowsy_pcts, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black', 
             label='Drowsy (KSS 8-9)')
    ax1.set_xlabel('Session Progress (%)', fontsize=11)
    ax1.set_ylabel('Sample Count', fontsize=11)
    ax1.set_title('When Does Drowsiness Occur?', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Comparison: drowsy vs alert distribution
    ax2 = axes[0, 1]
    bins = np.linspace(0, 100, 21)
    ax2.hist(all_alert_pcts, bins=bins, alpha=0.6, color='#3498db', 
             label='Alert', density=True)
    ax2.hist(all_drowsy_pcts, bins=bins, alpha=0.6, color='#e74c3c', 
             label='Drowsy', density=True)
    ax2.set_xlabel('Session Progress (%)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Alert vs Drowsy Temporal Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. First drowsy occurrence per subject
    ax3 = axes[1, 0]
    first_drowsy = [results[s]['first_drowsy_pct'] for s in results 
                    if results[s]['first_drowsy_pct'] is not None]
    ax3.hist(first_drowsy, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.axvline(np.median(first_drowsy), color='red', linestyle='--', linewidth=2,
               label=f'Median: {np.median(first_drowsy):.1f}%')
    ax3.set_xlabel('Session Progress (%)', fontsize=11)
    ax3.set_ylabel('Subject Count', fontsize=11)
    ax3.set_title('First Drowsy Episode Timing', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Drowsy ratio per subject
    ax4 = axes[1, 1]
    drowsy_ratios = [results[s]['drowsy_ratio'] * 100 for s in results]
    ax4.hist(drowsy_ratios, bins=20, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(drowsy_ratios), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(drowsy_ratios):.1f}%')
    ax4.set_xlabel('Drowsy Ratio (%)', fontsize=11)
    ax4.set_ylabel('Subject Count', fontsize=11)
    ax4.set_title('Drowsy Ratio Distribution Across Subjects', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Analysis of Drowsiness Occurrence', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    output_path = OUTPUT_DIR / "temporal_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_session_overview(results: Dict[str, Dict]) -> None:
    """Create an overview plot of all subjects showing session duration and drowsy timing."""
    subjects_sorted = sorted(results.keys(), 
                            key=lambda x: results[x]['drowsy_ratio'], 
                            reverse=True)
    
    fig, ax = plt.subplots(figsize=(16, max(8, len(subjects_sorted) * 0.18)))
    
    y_positions = np.arange(len(subjects_sorted))
    
    for i, subject_id in enumerate(subjects_sorted):
        data = results[subject_id]
        
        # Draw session bar
        ax.barh(i, 100, height=0.8, color='#ecf0f1', edgecolor='#bdc3c7')
        
        # Overlay drowsy segments
        drowsy_pcts = data['drowsy_time_pcts']
        if len(drowsy_pcts) > 0:
            # Find contiguous drowsy segments
            for pct in drowsy_pcts:
                ax.scatter(pct, i, color='#e74c3c', s=8, alpha=0.7, marker='|')
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(subjects_sorted, fontsize=7)
    ax.set_xlabel('Session Progress (%)', fontsize=12)
    ax.set_ylabel('Subject (sorted by drowsy ratio)', fontsize=12)
    ax.set_title('Session Overview: Drowsy Occurrence Timing\n(Red marks indicate drowsy samples)', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "session_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def save_summary_statistics(results: Dict[str, Dict]) -> None:
    """Save summary statistics to CSV."""
    rows = []
    for subject_id, data in results.items():
        rows.append({
            'subject_id': subject_id,
            'n_samples': data['n_samples'],
            'n_alert': data['n_alert'],
            'n_drowsy': data['n_drowsy'],
            'drowsy_ratio': data['drowsy_ratio'],
            'session_duration_sec': data['session_duration_sec'],
            'first_drowsy_pct': data['first_drowsy_pct'],
            'last_drowsy_pct': data['last_drowsy_pct'],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('drowsy_ratio', ascending=False)
    
    output_path = OUTPUT_DIR / "drowsy_summary.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"Saved: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("DROWSY TIMELINE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total subjects: {len(results)}")
    print(f"Total samples: {df['n_samples'].sum():,}")
    print(f"Total drowsy samples: {df['n_drowsy'].sum():,}")
    print(f"Total alert samples: {df['n_alert'].sum():,}")
    print(f"Overall drowsy ratio: {df['n_drowsy'].sum() / df['n_samples'].sum() * 100:.2f}%")
    print(f"\nSubjects with any drowsy samples: {(df['n_drowsy'] > 0).sum()}/{len(df)}")
    print(f"Mean drowsy ratio per subject: {df['drowsy_ratio'].mean() * 100:.2f}%")
    print(f"Median first drowsy occurrence: {df['first_drowsy_pct'].median():.1f}% of session")
    print("=" * 60)
    
    return df


def main():
    """Main function to run drowsy timeline visualization."""
    logging.info("Starting drowsy timeline analysis...")
    
    # Analyze all subjects
    results = analyze_all_subjects()
    logging.info(f"Analyzed {len(results)} subjects")
    
    # Generate visualizations
    logging.info("Generating individual timelines...")
    plot_individual_timelines(results, max_subjects=40)
    
    logging.info("Generating heatmap...")
    plot_aggregated_heatmap(results)
    
    logging.info("Generating temporal distribution...")
    plot_temporal_distribution(results)
    
    logging.info("Generating session overview...")
    plot_session_overview(results)
    
    # Save summary
    summary_df = save_summary_statistics(results)
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Generated files:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
