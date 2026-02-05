#!/usr/bin/env python3
"""
Generate Professional Learning Curves for Journals
==================================================
Includes: Accuracy/Loss, Compression Analysis, SNR, CV, and Confusion Matrices.
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def parse_run_name(name):
    """Extract parameters from run directory name."""
    info = {'seed': name.split("_seed")[-1]}
    
    if "arch_" in name:
        info['group'] = 'Architecture'
        parts = name.split("_seed")[0].split("_")
        for p in parts:
            if p.startswith("gr"): info['Growth Rate'] = int(p[2:])
            if p.startswith("c"): info['Compression'] = float(p[1:])
            if p.startswith("d"): info['Depth'] = p[1:].replace('_', '-')
            
    elif "batch_" in name:
        info['group'] = 'Batch Size'
        val = int(name.split("_")[1])
        info['Batch Size'] = val
        
    elif "res_" in name:
        info['group'] = 'Resolution'
        val = int(name.split("_")[1])
        info['Resolution'] = val
        
    return info

def plot_learning_curves_with_loss(df, group_df, figures_dir, x_col="Epoch"):
    """Generate Accuracy AND Loss FacetGrids."""
    
    # --- ACCURACY ---
    if 'Depth' in group_df.columns:
        depths = sorted(group_df['Depth'].unique())
        g = sns.relplot(
            data=group_df, x=x_col, y="Accuracy",
            hue="Growth Rate", col="Depth", col_order=depths,
            kind="line", palette="viridis", height=3.5, aspect=1.1, linewidth=2
        )
        g.fig.suptitle("Accuracy: Depth & Growth Rate", y=1.05, weight='bold')
        g.savefig(figures_dir / "journal_arch_accuracy.png", dpi=300, bbox_inches='tight')
        plt.close('all')
        
        # --- LOSS ---
        g = sns.relplot(
            data=group_df, x=x_col, y="Loss",
            hue="Growth Rate", col="Depth", col_order=depths,
            kind="line", palette="magma", height=3.5, aspect=1.1, linewidth=2
        )
        g.fig.suptitle("Loss: Depth & Growth Rate", y=1.05, weight='bold')
        g.savefig(figures_dir / "journal_arch_loss.png", dpi=300, bbox_inches='tight')
        plt.close('all')

def plot_compression_analysis(df, figures_dir):
    """Plot Compression Factor impact."""
    comp_df = df[df['group'] == 'Architecture']
    if comp_df.empty: return
    
    # Accuracy vs Compression
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=comp_df, x="Epoch", y="Accuracy", hue="Compression", 
                 palette="crest", linewidth=2.5)
    plt.title("Impact of Compression Factor on Accuracy", weight='bold')
    plt.savefig(figures_dir / "journal_compression_accuracy.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Loss vs Compression
    plt.figure(figsize=(7, 5))
    sns.lineplot(data=comp_df, x="Epoch", y="Loss", hue="Compression", 
                 palette="flare", linewidth=2.5)
    plt.title("Impact of Compression Factor on Loss", weight='bold')
    plt.savefig(figures_dir / "journal_compression_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_snr_robustness(results_dir, figures_dir):
    """Plot SNR Robustness from csv."""
    snr_file = results_dir / "snr_robustness/snr_results.csv"
    if not snr_file.exists():
        print(f"Skipping SNR: {snr_file} not found")
        return

    df = pd.read_csv(snr_file)
    print(f"Plotting SNR Analysis using {len(df)} records...")
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="snr_db", y="test_accuracy", marker='o', linewidth=3, color='#E74C3C')
    plt.title("Model Robustness to Signal-to-Noise Ratio (SNR)", weight='bold')
    plt.ylabel("Accuracy (%)")
    plt.xlabel("SNR (dB)")
    plt.grid(True, alpha=0.3)
    plt.fill_between(df["snr_db"], df["test_accuracy"] - 2, df["test_accuracy"] + 2, alpha=0.1, color='#E74C3C')
    plt.savefig(figures_dir / "journal_snr_robustness.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_cross_validation(results_dir, figures_dir):
    """Plot K-Fold CV Results."""
    cv_file = results_dir / "cross_validation/cv_results.csv"
    if not cv_file.exists():
        print("Skipping CV plot: File not found")
        return

    df = pd.read_csv(cv_file)
    # Expecting columns like 'fold', 'accuracy', 'val_loss'
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="fold", y="val_accuracy", palette="Blues_d", errorbar=None)
    plt.ylim(0, 100)
    plt.title("5-Fold Cross-Validation Performance", weight='bold')
    plt.ylabel("Validation Accuracy (%)")
    plt.savefig(figures_dir / "journal_cross_validation.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_plots():
    results_dir = Path(r"c:\Users\hkphi\Downloads\DRONE RFB SPECTRA\research\results")
    runs_dir = results_dir / "runs"
    figures_dir = results_dir / "figures_journal_v2"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Collecting updated data...")
    data_records = []
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir(): continue
        history_file = run_path / "history.csv"
        if not history_file.exists(): continue
        
        try:
            info = parse_run_name(run_path.name)
            df = pd.read_csv(history_file)
            for _, row in df.iterrows():
                record = info.copy()
                record['Epoch'] = row['epoch'] + 1
                record['Accuracy'] = row['val_accuracy'] * 100
                record['Loss'] = row['val_loss']
                data_records.append(record)
        except: pass

    full_df = pd.DataFrame(data_records)
    
    # PLOTTING SETUP
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({'font.family': 'serif'})
    
    # 1. Architecture Learning Curves (Acc + Loss)
    plot_learning_curves_with_loss(full_df, full_df[full_df['group'] == 'Architecture'], figures_dir)
    
    # 2. Compression Factor Analysis
    plot_compression_analysis(full_df, figures_dir)
    
    # 3. SNR Robustness
    plot_snr_robustness(results_dir, figures_dir)
    
    # 4. Cross Validation
    plot_cross_validation(results_dir, figures_dir)
    
    print(f"Saved V2 figures to: {figures_dir}")

if __name__ == "__main__":
    generate_plots()
