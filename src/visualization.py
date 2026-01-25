# -*- coding: utf-8 -*-
"""
Visualization Module for DroneRFB-Spectra Framework
====================================================

This module generates publication-quality visualizations for:
- Confusion matrices (normalized and absolute)
- Training curves (accuracy, loss, learning rate)
- Ablation study comparisons
- Radar charts for multi-metric comparison
- ROC curves and Precision-Recall curves
- Model comparison plots

All visualizations follow journal publication standards:
- 300 DPI minimum resolution
- LaTeX-compatible fonts
- Professional color schemes
- Clear legends and annotations
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Configure matplotlib for publication quality
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

logger = logging.getLogger(__name__)


# =============================================================================
# Color Schemes
# =============================================================================

# Professional color palette for publications
COLORS = {
    'primary': '#2563EB',      # Blue
    'secondary': '#10B981',    # Green
    'accent': '#F59E0B',       # Amber
    'error': '#EF4444',        # Red
    'neutral': '#6B7280',      # Gray
}

# Extended palette for multi-class plots
EXTENDED_PALETTE = sns.color_palette("husl", 25)


# =============================================================================
# Confusion Matrix Visualizations
# =============================================================================

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = None,
    cmap: str = 'Blues',
    show_values: bool = True,
    dpi: int = 300
) -> plt.Figure:
    """
    Create publication-quality confusion matrix visualization.
    
    Features:
    ---------
    - Adaptive sizing based on number of classes
    - Normalized or absolute values
    - Color-coded cells with value annotations
    - Clear axis labels with class names
    
    Args:
        confusion_matrix: NxN confusion matrix array
        class_labels: List of class names
        title: Plot title
        normalize: Normalize to show percentages
        save_path: Path to save figure
        figsize: Figure size tuple (width, height)
        cmap: Colormap name
        show_values: Show numeric values in cells
        dpi: Figure resolution
        
    Returns:
        Matplotlib Figure object
    """
    n_classes = len(class_labels)
    
    # Adaptive figure size
    if figsize is None:
        if n_classes <= 10:
            figsize = (8, 7)
        elif n_classes <= 20:
            figsize = (12, 10)
        else:
            figsize = (16, 14)
    
    # Normalize if requested
    cm = confusion_matrix.copy()
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        vmax = 1.0
    else:
        fmt = 'd'
        vmax = None
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmax=vmax)
    
    # Colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion' if normalize else 'Count')
    
    # Labels
    ax.set_title(title, fontweight='bold', pad=10)
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    
    # Tick labels
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.set_yticklabels(class_labels)
    
    # Add value annotations
    if show_values and n_classes <= 25:
        thresh = cm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                value = f"{cm[i, j]:{fmt}}" if not normalize else f"{cm[i, j]:.2f}"
                ax.text(j, i, value,
                       ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black',
                       fontsize=8 if n_classes <= 15 else 6)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


# =============================================================================
# Training History Visualizations
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Creates a 2-panel figure showing:
    - Left: Training/Validation Accuracy
    - Right: Training/Validation Loss
    
    Args:
        history: Training history dictionary with keys 'accuracy', 'val_accuracy', 'loss', 'val_loss'
        title: Overall figure title
        save_path: Path to save figure
        dpi: Figure resolution
        
    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=dpi)
    
    epochs = range(1, len(history.get('accuracy', history.get('loss', []))) + 1)
    
    # Accuracy plot
    ax = axes[0]
    if 'accuracy' in history:
        ax.plot(epochs, history['accuracy'], 
               color=COLORS['primary'], linestyle='-', linewidth=2,
               label='Training', marker='o', markersize=3)
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], 
               color=COLORS['secondary'], linestyle='--', linewidth=2,
               label='Validation', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy', fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim([0, 1.05])
    
    # Loss plot
    ax = axes[1]
    if 'loss' in history:
        ax.plot(epochs, history['loss'], 
               color=COLORS['primary'], linestyle='-', linewidth=2,
               label='Training', marker='o', markersize=3)
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], 
               color=COLORS['secondary'], linestyle='--', linewidth=2,
               label='Validation', marker='s', markersize=3)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.suptitle(title, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
    
    return fig


# =============================================================================
# Radar Chart for Multi-Metric Comparison
# =============================================================================

def plot_radar_chart(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: List[str] = None,
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create radar chart for multi-metric model comparison.
    
    Radar charts are effective for comparing models across multiple
    dimensions simultaneously, making trade-offs visually apparent.
    
    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries
        metric_names: List of metric names to include
        title: Chart title
        save_path: Path to save figure
        dpi: Figure resolution
        
    Returns:
        Matplotlib Figure object
        
    Example:
        >>> metrics = {
        ...     'RF-DenseNet': {'Accuracy': 0.96, 'F1': 0.95, 'Latency': 0.9},
        ...     'VGG16': {'Accuracy': 0.83, 'F1': 0.82, 'Latency': 0.3}
        ... }
        >>> plot_radar_chart(metrics)
    """
    if metric_names is None:
        # Get common metrics from first model
        first_model = next(iter(metrics_dict.values()))
        metric_names = list(first_model.keys())
    
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=dpi)
    
    colors = EXTENDED_PALETTE[:len(metrics_dict)]
    
    for idx, (model_name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) for m in metric_names]
        values += values[:1]  # Close the polygon
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, size=10)
    ax.set_ylim(0, 1.1)
    
    ax.set_title(title, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
        logger.info(f"Radar chart saved to {save_path}")
    
    return fig


# =============================================================================
# ROC and Precision-Recall Curves
# =============================================================================

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_labels: List[str],
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Uses One-vs-Rest strategy to compute ROC curve for each class.
    Includes micro and macro-averaged curves for overall performance.
    
    Args:
        y_true: True labels (integer encoded)
        y_prob: Prediction probabilities of shape (N, num_classes)
        class_labels: List of class names
        title: Plot title
        save_path: Path to save figure
        dpi: Figure resolution
        
    Returns:
        Matplotlib Figure object
    """
    n_classes = len(class_labels)
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    # Compute ROC curve for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    # Compute macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    
    # Plot macro-average
    ax.plot(fpr['macro'], tpr['macro'],
           label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
           color=COLORS['primary'], linestyle='-', linewidth=3)
    
    # Plot micro-average
    ax.plot(fpr['micro'], tpr['micro'],
           label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
           color=COLORS['secondary'], linestyle='--', linewidth=3)
    
    # Plot per-class if not too many
    if n_classes <= 10:
        colors = EXTENDED_PALETTE[:n_classes]
        for i, color in enumerate(colors):
            ax.plot(fpr[i], tpr[i], color=color, alpha=0.6, linewidth=1,
                   label=f'{class_labels[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
    
    return fig


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_labels: List[str],
    title: str = "Precision-Recall Curves",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot precision-recall curves for multi-class classification.
    
    Precision-recall curves are particularly useful for imbalanced
    datasets where ROC curves may be overly optimistic.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        class_labels: Class names
        title: Plot title
        save_path: Save path
        dpi: Resolution
        
    Returns:
        Matplotlib Figure object
    """
    n_classes = len(class_labels)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    
    colors = EXTENDED_PALETTE[:n_classes] if n_classes <= 10 else [COLORS['primary']]
    
    avg_precisions = []
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        avg_precisions.append(ap)
        
        if n_classes <= 10:
            ax.plot(recall, precision, color=colors[i], linewidth=1.5, alpha=0.7,
                   label=f'{class_labels[i]} (AP = {ap:.3f})')
    
    mean_ap = np.mean(avg_precisions)
    ax.axhline(y=mean_ap, color=COLORS['accent'], linestyle='--', linewidth=2,
              label=f'Mean AP = {mean_ap:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
        logger.info(f"PR curves saved to {save_path}")
    
    return fig


# =============================================================================
# Ablation Study Visualizations
# =============================================================================

def plot_ablation_study(
    ablation_df: pd.DataFrame,
    x_col: str,
    y_col: str = 'accuracy',
    hue_col: Optional[str] = None,
    title: str = "Ablation Study",
    xlabel: str = None,
    ylabel: str = "Accuracy",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create ablation study visualization.
    
    Args:
        ablation_df: DataFrame with ablation results
        x_col: Column for x-axis
        y_col: Column for y-axis (metric)
        hue_col: Column for grouping/coloring
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Save path
        dpi: Resolution
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    
    if hue_col:
        for name, group in ablation_df.groupby(hue_col):
            ax.plot(group[x_col], group[y_col], 'o-', linewidth=2, 
                   markersize=8, label=f'{hue_col}={name}')
    else:
        ax.plot(ablation_df[x_col], ablation_df[y_col], 'o-', 
               linewidth=2, markersize=8, color=COLORS['primary'])
    
    ax.set_xlabel(xlabel or x_col, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    if hue_col:
        ax.legend()
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    metric_col: str = 'accuracy',
    model_col: str = 'model',
    title: str = "Model Comparison",
    save_path: Optional[Path] = None,
    dpi: int = 300,
    highlight_best: bool = True
) -> plt.Figure:
    """
    Create bar chart comparing models on a metric.
    
    Args:
        comparison_df: DataFrame with model comparison results
        metric_col: Column with metric values
        model_col: Column with model names
        title: Plot title
        save_path: Save path
        dpi: Resolution
        highlight_best: Highlight the best performing model
        
    Returns:
        Matplotlib Figure object
    """
    df = comparison_df.sort_values(metric_col, ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    colors = [COLORS['primary']] * len(df)
    if highlight_best:
        colors[0] = COLORS['secondary']  # Best model in green
    
    bars = ax.barh(df[model_col], df[metric_col], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, df[metric_col]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.4f}', va='center', fontsize=9)
    
    ax.set_xlabel(metric_col.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0, df[metric_col].max() * 1.15])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_accuracy_vs_latency(
    comparison_df: pd.DataFrame,
    accuracy_col: str = 'accuracy',
    latency_col: str = 'avg_latency_ms',
    model_col: str = 'model',
    params_col: str = 'total_params',
    title: str = "Accuracy vs. Inference Latency",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create scatter plot of accuracy vs. latency with model size.
    
    This visualization helps identify the Pareto frontier of
    accuracy-efficiency trade-offs.
    
    Args:
        comparison_df: DataFrame with comparison data
        accuracy_col: Accuracy column
        latency_col: Latency column
        model_col: Model name column
        params_col: Parameter count column
        title: Plot title
        save_path: Save path
        dpi: Resolution
        
    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)
    
    # Size based on parameter count (normalized)
    if params_col in comparison_df.columns:
        params = comparison_df[params_col]
        sizes = 100 + (params - params.min()) / (params.max() - params.min()) * 500
    else:
        sizes = 200
    
    scatter = ax.scatter(
        comparison_df[latency_col],
        comparison_df[accuracy_col],
        s=sizes,
        c=range(len(comparison_df)),
        cmap='viridis',
        alpha=0.7,
        edgecolors='black',
        linewidth=1
    )
    
    # Add model labels
    for idx, row in comparison_df.iterrows():
        ax.annotate(
            row[model_col],
            (row[latency_col], row[accuracy_col]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    ax.set_xlabel('Inference Latency (ms)', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble size
    if params_col in comparison_df.columns:
        size_legend = ax.legend(
            *scatter.legend_elements(prop="sizes", num=4, alpha=0.6),
            title="Parameters",
            loc="lower right"
        )
        ax.add_artist(size_legend)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
    
    return fig


# =============================================================================
# Dataset Visualization
# =============================================================================

def plot_class_distribution(
    class_counts: Dict[str, int],
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot dataset class distribution as horizontal bar chart.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        title: Plot title
        save_path: Save path
        dpi: Resolution
        
    Returns:
        Matplotlib Figure object
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.3)), dpi=dpi)
    
    colors = [COLORS['primary']] * len(classes)
    bars = ax.barh(classes, counts, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(count + max(counts) * 0.01, bar.get_y() + bar.get_height()/2,
               f'{count}', va='center', fontsize=9)
    
    ax.set_xlabel('Number of Samples', fontweight='bold')
    ax.set_ylabel('Class', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim([0, max(counts) * 1.15])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add summary statistics
    total = sum(counts)
    mean = np.mean(counts)
    std = np.std(counts)
    ax.text(0.95, 0.05, f'Total: {total:,}\nMean: {mean:.1f}\nStd: {std:.1f}',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        for ext in ['png', 'pdf']:
            fig.savefig(save_path.with_suffix(f'.{ext}'), dpi=dpi, bbox_inches='tight')
    
    return fig


def close_all_figures():
    """Close all matplotlib figures to free memory."""
    plt.close('all')
