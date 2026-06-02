"""
Visualization tools for VegeSSL results.

Generates analysis plots for isolation scores and detected errors.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import load_config


# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def save_figure(fig, name, output_dir):
    """Save figure in multiple formats."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f"{name}.pdf"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {name}.png and .pdf")


def plot_score_histogram(df, thresholds, output_dir):
    """Plot isolation score distribution with threshold levels."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    scores = df['isolation_score'].values
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    for (level_name, color) in zip(['level1', 'level2', 'level3', 'level4'], colors):
        level = getattr(thresholds, level_name)
        ax.axvline(x=level.value, color=color, linestyle='--', linewidth=2.5,
                   label=f"{level.name}: {level.value}")
    
    ax.axvline(x=scores.mean(), color='black', linestyle='-', linewidth=2,
               label=f'Mean: {scores.mean():.3f}')
    
    ax.set_xlabel('Isolation Score')
    ax.set_ylabel('Number of Segments')
    ax.set_title('Distribution of Segment Isolation Scores')
    ax.legend(loc='upper right')
    
    textstr = f'N={len(scores):,}\nMean={scores.mean():.3f}\nStd={scores.std():.3f}'
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure(fig, "isolation_score_histogram", output_dir)


def plot_class_boxplot(df, thresholds, error_colors, output_dir):
    """Plot class-wise isolation score distribution."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    class_names = df['class_name'].unique()
    class_data = []
    class_labels = []
    box_colors = []
    
    for cls_name in class_names:
        mask = df['class_name'] == cls_name
        if mask.sum() > 0:
            class_data.append(df.loc[mask, 'isolation_score'].values)
            class_labels.append(cls_name)
            
            cls_idx = df.loc[mask, 'class_idx'].iloc[0]
            color = error_colors.get(str(cls_idx), [100, 100, 100])
            box_colors.append(np.array(color) / 255)
    
    bp = ax.boxplot(class_data, patch_artist=True, labels=class_labels)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    threshold_colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    for (level_name, color) in zip(['level1', 'level2', 'level3', 'level4'], threshold_colors):
        level = getattr(thresholds, level_name)
        ax.axhline(y=level.value, color=color, linestyle='--', linewidth=1.5,
                   label=f"{level.name}: {level.value}", alpha=0.7)
    
    ax.set_xlabel('Vegetation Class')
    ax.set_ylabel('Isolation Score')
    ax.set_title('Isolation Score Distribution by Class')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)
    
    save_figure(fig, "class_boxplot", output_dir)


def plot_threshold_comparison(df, thresholds, output_dir):
    """Plot detection count at each threshold level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = df['isolation_score'].values
    colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    
    level_counts = []
    level_labels = []
    
    for level_name in ['level1', 'level2', 'level3', 'level4']:
        level = getattr(thresholds, level_name)
        count = (scores >= level.value).sum()
        level_counts.append(count)
        level_labels.append(f"{level.name}\n≥{level.value}")
    
    bars = ax.bar(level_labels, level_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Suspicious Segments')
    ax.set_title('Detection Count by Threshold Level')
    
    for bar, count in zip(bars, level_counts):
        pct = count / len(scores) * 100
        ax.text(bar.get_x() + bar.get_width()/2, count + max(level_counts)*0.02,
                f'{count:,}\n({pct:.1f}%)', ha='center', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    save_figure(fig, "threshold_comparison", output_dir)


def plot_area_vs_score(df, thresholds, error_colors, output_dir):
    """Plot segment area vs isolation score."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = df['isolation_score'].values
    areas = df['area_pixels'].values
    
    colors = []
    for _, row in df.iterrows():
        cls_idx = row['class_idx']
        color = error_colors.get(str(cls_idx), [100, 100, 100])
        colors.append(np.array(color) / 255)
    
    ax.scatter(np.log10(areas + 1), scores, c=colors, alpha=0.5, s=20)
    
    threshold_colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    for (level_name, color) in zip(['level1', 'level2', 'level3', 'level4'], threshold_colors):
        level = getattr(thresholds, level_name)
        ax.axhline(y=level.value, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('log₁₀(Segment Area + 1)')
    ax.set_ylabel('Isolation Score')
    ax.set_title('Segment Size vs Isolation Score')
    
    corr = np.corrcoef(np.log10(areas + 1), scores)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: r = {corr:.3f}', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    save_figure(fig, "area_vs_score", output_dir)


def plot_cumulative_distribution(df, thresholds, output_dir):
    """Plot cumulative detection curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scores = df['isolation_score'].values
    sorted_scores = np.sort(scores)[::-1]
    cumulative = np.arange(1, len(sorted_scores) + 1)
    
    ax.plot(sorted_scores, cumulative, 'b-', linewidth=2)
    
    colors = ['#27ae60', '#f39c12', '#e74c3c', '#8e44ad']
    for (level_name, color) in zip(['level1', 'level2', 'level3', 'level4'], colors):
        level = getattr(thresholds, level_name)
        n_above = (scores >= level.value).sum()
        ax.axvline(x=level.value, color=color, linestyle='--', linewidth=2)
        ax.scatter([level.value], [n_above], color=color, s=100, zorder=5, edgecolors='black')
        ax.annotate(f'{level.name}: {n_above:,}', xy=(level.value, n_above),
                    xytext=(level.value + 0.05, n_above * 1.1), fontsize=9, color=color)
    
    ax.set_xlabel('Isolation Score')
    ax.set_ylabel('Cumulative Count')
    ax.set_title('Cumulative Distribution (Error Detection Curve)')
    
    save_figure(fig, "cumulative_distribution", output_dir)


def main(args):
    config = load_config(args.config)
    
    # Load analysis results
    csv_path = os.path.join(config.output.tables_dir, "all_segments_analysis.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Analysis results not found: {csv_path}\nRun detect_errors.py first.")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} segments")
    
    output_dir = config.output.figures_dir
    thresholds = config.detection.thresholds
    error_colors = config.visualization.error_colors
    
    # Convert error_colors to dict if needed
    if hasattr(error_colors, 'to_dict'):
        error_colors = error_colors.to_dict()
    
    print("Generating visualizations...")
    
    plot_score_histogram(df, thresholds, output_dir)
    plot_class_boxplot(df, thresholds, error_colors, output_dir)
    plot_threshold_comparison(df, thresholds, output_dir)
    plot_area_vs_score(df, thresholds, error_colors, output_dir)
    plot_cumulative_distribution(df, thresholds, output_dir)
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate analysis visualizations")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args)
