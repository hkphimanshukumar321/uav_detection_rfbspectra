#!/usr/bin/env python3
"""
Full Factorial Ablation Study with Statistical Significance
============================================================

Features:
- Full factorial design: GR × Compression × Depth (27) + Batch (3) + Resolution (3) = 33 configs
- Multiple seeds per config (3 seeds) for statistical significance → 99 total experiments
- Multi-GPU support with tf.distribute.MirroredStrategy
- Combined accuracy/loss plots per ablation parameter
- Machine-specific inference time logging
- All journal-ready visualizations (ROC, PR, confusion matrix)
"""

import sys
import time
import socket
import platform
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from itertools import product
from typing import List, Dict, Tuple, Optional

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ResearchConfig, print_experiment_summary
from src.training import (
    train_model, setup_gpu, compile_model, compute_metrics, 
    generate_run_id, benchmark_inference, get_device_info
)
from src.models import create_rf_densenet, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset
from src.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    plot_ablation_study,
    plot_model_comparison_bar,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_radar_chart,
    plot_accuracy_vs_latency,
    close_all_figures
)


def get_machine_info() -> Dict:
    """Get machine-specific info for reproducibility."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'num_gpus': len(gpus),
        'gpu_names': [gpu.name for gpu in gpus],
        'timestamp': datetime.now().isoformat()
    }


def setup_multi_gpu():
    """Setup multi-GPU training strategy."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        # Multi-GPU: use MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ Multi-GPU enabled: {len(gpus)} GPUs detected")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        return strategy
    elif len(gpus) == 1:
        # Single GPU
        print(f"✓ Single GPU detected: {gpus[0].name}")
        return tf.distribute.get_strategy()  # Default strategy
    else:
        # CPU only
        print("⚠ No GPU detected, using CPU")
        return tf.distribute.get_strategy()


class AblationProgress:
    """Track ablation study progress with ETA."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.start = time.time()
        self.category = ""
    
    def set_category(self, cat: str):
        self.category = cat
    
    def update(self):
        self.completed += 1
        pct = self.completed / self.total
        elapsed = time.time() - self.start
        eta = (elapsed / self.completed) * (self.total - self.completed) if self.completed > 0 else 0
        eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
        bar = "█" * int(40 * pct) + "░" * (40 - int(40 * pct))
        print(f"\r[{bar}] {pct*100:5.1f}% | {self.completed}/{self.total} | ETA: {eta_str} | {self.category}", 
              end="", flush=True)
        if self.completed == self.total:
            print()


def run_ablation(quick_test: bool = False, single_seed: bool = False):
    """
    Run full factorial ablation study.
    
    Args:
        quick_test: Use 2 epochs per experiment
        single_seed: Use only first seed (for quick testing)
    """
    print("=" * 70)
    print("FULL FACTORIAL ABLATION STUDY")
    print("With Statistical Significance (Multiple Seeds)")
    print("=" * 70)
    
    config = ResearchConfig()
    results_dir = config.output.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 2 if quick_test else config.training.epochs
    # Use config flag OR override with single_seed parameter
    use_single = single_seed or (not config.training.use_multiple_seeds)
    seeds = [config.training.seeds[0]] if use_single else config.training.seeds
    
    # =========================================================================
    # PHASE 1: Setup & Machine Info
    # =========================================================================
    print("\n[1/5] Environment Setup...")
    
    machine_info = get_machine_info()
    print(f"  Hostname: {machine_info['hostname']}")
    print(f"  Platform: {machine_info['platform']}")
    print(f"  TensorFlow: {machine_info['tensorflow_version']}")
    print(f"  GPUs: {machine_info['num_gpus']}")
    
    strategy = setup_multi_gpu()
    
    # Save machine info
    machine_log = results_dir / "machine_info.json"
    import json
    with open(machine_log, 'w') as f:
        json.dump(machine_info, f, indent=2)
    print(f"  ✓ Machine info saved: {machine_log}")
    
    # =========================================================================
    # PHASE 2: Load Data
    # =========================================================================
    print("\n[2/5] Loading Dataset...")
    
    data_dir = config.data.data_dir
    categories, _ = validate_dataset_directory(data_dir, min_classes=2)
    
    X, Y = load_dataset_numpy(
        data_dir=data_dir,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )
    print(f"  Loaded: {len(X)} images, shape: {X.shape}")
    
    num_classes = len(categories)
    input_shape = (config.data.img_size[0], config.data.img_size[1], 3)
    
    # =========================================================================
    # PHASE 3: Calculate Experiment Design
    # =========================================================================
    print("\n[3/5] Experiment Design...")
    
    arch_combos = list(product(
        config.ablation.growth_rates,
        config.ablation.compressions,
        config.ablation.depths
    ))
    n_arch = len(arch_combos)
    n_batch = len(config.ablation.batch_sizes)
    n_resolution = len(config.ablation.resolutions)
    n_seeds = len(seeds)
    
    n_configs = n_arch + n_batch + n_resolution
    total_experiments = n_configs * n_seeds
    
    print(f"\n📊 ABLATION DESIGN:")
    print(f"   Architecture (GR × Comp × Depth): {n_arch} configs")
    print(f"   Batch Size: {n_batch} configs")
    print(f"   Resolution: {n_resolution} configs")
    print(f"   Seeds per config: {n_seeds} {seeds}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL EXPERIMENTS: {n_configs} × {n_seeds} = {total_experiments}")
    print(f"   Epochs: {epochs}")
    
    progress = AblationProgress(total_experiments)
    all_results = []
    all_histories = {}  # For combined plots
    start_time = time.time()
    
    # =========================================================================
    # PHASE 4: Run Experiments
    # =========================================================================
    print("\n[4/5] Running Experiments...")
    print("=" * 70)
    
    # Helper function for single experiment
    def run_single(exp_id, gr, comp, depth, batch, res, seed):
        from sklearn.metrics import confusion_matrix as sk_cm, f1_score
        import tensorflow as tf
        
        # Set seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
        
        # Load data at correct resolution
        if res != config.data.img_size[0]:
            X_exp, Y_exp = load_dataset_numpy(
                data_dir=data_dir, categories=categories,
                img_size=(res, res), max_images_per_class=config.data.max_images_per_class,
                show_progress=False
            )
        else:
            X_exp, Y_exp = X, Y
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
            X_exp, Y_exp, test_size=0.15, val_size=0.15, seed=seed
        )
        
        # Calculate class weights for imbalance handling
        from sklearn.utils import class_weight
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights_vals))
        
        inp_shape = (res, res, 3)
        
        # Create model (within strategy scope for multi-GPU)
        with strategy.scope():
            model = create_rf_densenet(
                input_shape=inp_shape,
                num_classes=num_classes,
                growth_rate=gr,
                compression=comp,
                depth=depth,
                dropout_rate=config.model.dropout_rate,
                initial_filters=config.model.initial_filters
            )
            model = compile_model(model, learning_rate=config.training.learning_rate)
        
        metrics_info = get_model_metrics(model)
        
        # Train
        run_dir = results_dir / "runs" / f"{exp_id}_seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=batch,
            class_weights=class_weights,
            early_stopping_patience=config.training.early_stopping_patience
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Inference time (machine-specific)
        latency = benchmark_inference(model, inp_shape, warmup_runs=5, benchmark_runs=20)
        
        # NOTE: Individual plots removed - only combined summary plots are generated
        # This reduces image count from ~100 to ~10 for cleaner output
        close_all_figures()
        
        # Calculate edge device metrics
        model_size_kb = (metrics_info.total_params * 4) / 1024  # float32 = 4 bytes
        # Estimate memory: model + gradients + activations (rough: 3x model size)
        memory_mb = (model_size_kb * 3) / 1024
        
        # Cleanup to prevent OOM
        tf.keras.backend.clear_session()
        
        return {
            'experiment_id': exp_id,
            'seed': seed,
            'growth_rate': gr,
            'compression': comp,
            'depth': str(depth),
            'batch_size': batch,
            'resolution': res,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history.get('val_accuracy', [0])) * 100,
            # Edge device metrics
            'total_params': metrics_info.total_params,
            'model_size_kb': model_size_kb,
            'memory_mb': memory_mb,
            'inference_ms': latency['batch_1']['mean_ms'],
            'throughput_fps': latency['batch_1']['throughput_fps'],
            'machine': machine_info['hostname']
        }, history
    
    # ----- ARCHITECTURE ABLATION -----
    print("\n" + "─" * 70)
    print("GROUP 1: ARCHITECTURE (Growth Rate × Compression × Depth)")
    print("─" * 70)
    
    arch_histories = []
    for idx, (gr, comp, depth) in enumerate(arch_combos, 1):
        depth_str = '_'.join(map(str, depth))
        exp_id = f"arch_gr{gr}_c{comp}_d{depth_str}"
        
        for seed in seeds:
            progress.set_category(f"ARCH: GR={gr},C={comp},D={depth} [seed={seed}]")
            result, history = run_single(exp_id, gr, comp, depth, 
                                          config.training.batch_size, 
                                          config.data.img_size[0], seed)
            result['ablation_group'] = 'architecture'
            all_results.append(result)
            # Intermediate save
            pd.DataFrame([result]).to_csv(results_dir / "ablation_full_factorial.csv", 
                                        mode='a', header=not (results_dir / "ablation_full_factorial.csv").exists(), 
                                        index=False)
            arch_histories.append({'exp_id': exp_id, 'seed': seed, 'history': history})
            progress.update()
    
    # ----- BATCH SIZE ABLATION -----
    print("\n" + "─" * 70)
    print("GROUP 2: BATCH SIZE")
    print("─" * 70)
    
    for bs in config.ablation.batch_sizes:
        exp_id = f"batch_{bs}"
        for seed in seeds:
            progress.set_category(f"BATCH: {bs} [seed={seed}]")
            result, history = run_single(exp_id, config.model.growth_rate, 
                                          config.model.compression, config.model.depth,
                                          bs, config.data.img_size[0], seed)
            result['ablation_group'] = 'batch_size'
            all_results.append(result)
            # Intermediate save
            pd.DataFrame([result]).to_csv(results_dir / "ablation_full_factorial.csv", 
                                        mode='a', header=not (results_dir / "ablation_full_factorial.csv").exists(), 
                                        index=False)
            progress.update()
    
    # ----- RESOLUTION ABLATION -----
    print("\n" + "─" * 70)
    print("GROUP 3: RESOLUTION")
    print("─" * 70)
    
    for res in config.ablation.resolutions:
        exp_id = f"res_{res}"
        for seed in seeds:
            progress.set_category(f"RESOLUTION: {res}×{res} [seed={seed}]")
            result, history = run_single(exp_id, config.model.growth_rate,
                                          config.model.compression, config.model.depth,
                                          config.training.batch_size, res, seed)
            result['ablation_group'] = 'resolution'
            all_results.append(result)
            # Intermediate save
            pd.DataFrame([result]).to_csv(results_dir / "ablation_full_factorial.csv", 
                                        mode='a', header=not (results_dir / "ablation_full_factorial.csv").exists(), 
                                        index=False)
            progress.update()
    
    # =========================================================================
    # PHASE 5: Save Results & Generate Summary
    # =========================================================================
    print("\n\n[5/5] Saving Results & Summary Plots...")
    print("=" * 70)
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_csv = results_dir / "ablation_full_factorial.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved: {results_csv}")
    
    # Compute mean ± std per experiment_id
    summary_df = results_df.groupby('experiment_id').agg({
        'test_accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'inference_ms': ['mean', 'std'],
        'total_params': 'first',
        'ablation_group': 'first'
    }).reset_index()
    summary_df.columns = ['experiment_id', 'accuracy_mean', 'accuracy_std', 
                           'f1_mean', 'f1_std', 'latency_mean', 'latency_std',
                           'params', 'group']
    summary_csv = results_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Saved: {summary_csv}")
    
    # Generate combined plots per ablation group
    _generate_combined_plots(results_df, figures_dir)
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Duration: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"📊 Experiments: {len(all_results)} ({n_configs} configs × {n_seeds} seeds)")
    print(f"📈 Best Accuracy: {results_df['test_accuracy'].max():.2f}%")
    print(f"🖥️  Machine: {machine_info['hostname']}")
    print(f"📁 Results: {results_dir}")
    print("=" * 70)
    
    return results_df


def _generate_combined_plots(df: pd.DataFrame, figures_dir: Path):
    """Generate journal-quality combined plots for ablation study."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # =========================================================================
    # JOURNAL-QUALITY STYLE SETTINGS
    # =========================================================================
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Professional typography
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Color palette (colorblind-friendly)
    COLORS = {
        'primary': '#2E86AB',
        'secondary': '#A23B72', 
        'tertiary': '#F18F01',
        'quaternary': '#C73E1D',
        'architecture': '#2E86AB',
        'batch_size': '#28A745',
        'resolution': '#F18F01'
    }
    
    print("Generating journal-quality plots...")
    
    # =========================================================================
    # 1. ACCURACY + LOSS: Growth Rate (Side-by-side)
    # =========================================================================
    arch_df = df[df['ablation_group'] == 'architecture']
    if len(arch_df) > 0:
        gr_acc = arch_df.groupby('growth_rate')['test_accuracy'].agg(['mean', 'std']).reset_index()
        gr_loss = arch_df.groupby('growth_rate')['test_loss'].agg(['mean', 'std']).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy
        axes[0].errorbar(gr_acc['growth_rate'], gr_acc['mean'], yerr=gr_acc['std'], 
                         fmt='o-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['primary'], capthick=1.5)
        axes[0].set_xlabel('Growth Rate')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('(a) Accuracy vs Growth Rate')
        
        # Loss
        axes[1].errorbar(gr_loss['growth_rate'], gr_loss['mean'], yerr=gr_loss['std'], 
                         fmt='s-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['secondary'], capthick=1.5)
        axes[1].set_xlabel('Growth Rate')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('(b) Loss vs Growth Rate')
        
        fig.suptitle('Effect of Growth Rate on Model Performance', fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(figures_dir / 'growth_rate_acc_loss.png')
        plt.close(fig)
        
        # Compression
        comp_acc = arch_df.groupby('compression')['test_accuracy'].agg(['mean', 'std']).reset_index()
        comp_loss = arch_df.groupby('compression')['test_loss'].agg(['mean', 'std']).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].errorbar(comp_acc['compression'], comp_acc['mean'], yerr=comp_acc['std'], 
                         fmt='o-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['primary'], capthick=1.5)
        axes[0].set_xlabel('Compression Factor')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('(a) Accuracy vs Compression')
        
        axes[1].errorbar(comp_loss['compression'], comp_loss['mean'], yerr=comp_loss['std'], 
                         fmt='s-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['secondary'], capthick=1.5)
        axes[1].set_xlabel('Compression Factor')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('(b) Loss vs Compression')
        
        fig.suptitle('Effect of Compression on Model Performance', fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(figures_dir / 'compression_acc_loss.png')
        plt.close(fig)
    
    # =========================================================================
    # 2. BATCH SIZE: Accuracy + Loss
    # =========================================================================
    batch_df = df[df['ablation_group'] == 'batch_size']
    if len(batch_df) > 0:
        bs_acc = batch_df.groupby('batch_size')['test_accuracy'].agg(['mean', 'std']).reset_index()
        bs_loss = batch_df.groupby('batch_size')['test_loss'].agg(['mean', 'std']).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].errorbar(bs_acc['batch_size'], bs_acc['mean'], yerr=bs_acc['std'], 
                         fmt='^-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['batch_size'], capthick=1.5)
        axes[0].set_xlabel('Batch Size')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('(a) Accuracy vs Batch Size')
        
        axes[1].errorbar(bs_loss['batch_size'], bs_loss['mean'], yerr=bs_loss['std'], 
                         fmt='v-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['secondary'], capthick=1.5)
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('(b) Loss vs Batch Size')
        
        fig.suptitle('Effect of Batch Size on Model Performance', fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(figures_dir / 'batch_size_acc_loss.png')
        plt.close(fig)
    
    # =========================================================================
    # 3. RESOLUTION: Accuracy + Loss
    # =========================================================================
    res_df = df[df['ablation_group'] == 'resolution']
    if len(res_df) > 0:
        res_acc = res_df.groupby('resolution')['test_accuracy'].agg(['mean', 'std']).reset_index()
        res_loss = res_df.groupby('resolution')['test_loss'].agg(['mean', 'std']).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].errorbar(res_acc['resolution'], res_acc['mean'], yerr=res_acc['std'], 
                         fmt='D-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['resolution'], capthick=1.5)
        axes[0].set_xlabel('Input Resolution (pixels)')
        axes[0].set_ylabel('Test Accuracy (%)')
        axes[0].set_title('(a) Accuracy vs Resolution')
        
        axes[1].errorbar(res_loss['resolution'], res_loss['mean'], yerr=res_loss['std'], 
                         fmt='d-', capsize=4, linewidth=2, markersize=8, 
                         color=COLORS['secondary'], capthick=1.5)
        axes[1].set_xlabel('Input Resolution (pixels)')
        axes[1].set_ylabel('Test Loss')
        axes[1].set_title('(b) Loss vs Resolution')
        
        fig.suptitle('Effect of Input Resolution on Model Performance', fontweight='bold', y=1.02)
        plt.tight_layout()
        fig.savefig(figures_dir / 'resolution_acc_loss.png')
        plt.close(fig)
    
    # =========================================================================
    # 4. PARETO FRONT: Accuracy vs Latency (Edge Device Trade-off)
    # =========================================================================
    if len(df) > 0 and 'inference_ms' in df.columns:
        summary = df.groupby('experiment_id').agg({
            'test_accuracy': 'mean',
            'inference_ms': 'mean',
            'model_size_kb': 'first',
            'ablation_group': 'first'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        markers = {'architecture': 'o', 'batch_size': '^', 'resolution': 'D'}
        
        for group in summary['ablation_group'].unique():
            group_df = summary[summary['ablation_group'] == group]
            ax.scatter(group_df['inference_ms'], group_df['test_accuracy'], 
                       label=group.replace('_', ' ').title(), 
                       s=120, alpha=0.8, c=COLORS.get(group, 'gray'),
                       marker=markers.get(group, 'o'), edgecolors='white', linewidth=1)
        
        # Annotate best models
        best_idx = summary['test_accuracy'].idxmax()
        fastest_idx = summary['inference_ms'].idxmin()
        
        ax.annotate(f"Best ({summary.loc[best_idx, 'test_accuracy']:.1f}%)", 
                    xy=(summary.loc[best_idx, 'inference_ms'], summary.loc[best_idx, 'test_accuracy']),
                    xytext=(15, 10), textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        ax.annotate(f"Fastest ({summary.loc[fastest_idx, 'inference_ms']:.1f}ms)", 
                    xy=(summary.loc[fastest_idx, 'inference_ms'], summary.loc[fastest_idx, 'test_accuracy']),
                    xytext=(15, -15), textcoords='offset points', fontsize=9,
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        
        ax.set_xlabel('Inference Latency (ms)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Pareto Front: Accuracy vs Latency Trade-off', fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray')
        fig.savefig(figures_dir / 'pareto_accuracy_latency.png')
        plt.close(fig)
    
    # =========================================================================
    # 5. RADAR CHART: Top 5 Models Comparison
    # =========================================================================
    if len(df) > 0:
        summary = df.groupby('experiment_id').agg({
            'test_accuracy': 'mean',
            'macro_f1': 'mean',
            'inference_ms': 'mean',
            'model_size_kb': 'first',
            'throughput_fps': 'mean'
        }).reset_index()
        
        top5 = summary.nlargest(5, 'test_accuracy')
        categories = ['Accuracy', 'F1 Score', 'Speed', 'Compactness', 'Throughput']
        n_cats = len(categories)
        
        fig, ax = plt.subplots(figsize=(10, 9), subplot_kw=dict(projection='polar'))
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]
        
        colors_radar = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#6C757D']
        
        for i, (_, row) in enumerate(top5.iterrows()):
            values = [
                row['test_accuracy'] / 100,
                row['macro_f1'] / 100,
                1 - (row['inference_ms'] / summary['inference_ms'].max()),
                1 - (row['model_size_kb'] / summary['model_size_kb'].max()),
                row['throughput_fps'] / summary['throughput_fps'].max()
            ]
            values += values[:1]
            
            # Shorten label
            label = row['experiment_id'].replace('arch_', '').replace('_', ' ')[:18]
            ax.plot(angles, values, 'o-', linewidth=2, color=colors_radar[i], label=label)
            ax.fill(angles, values, alpha=0.15, color=colors_radar[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('Top 5 Models: Multi-Metric Comparison', fontweight='bold', y=1.08)
        
        # Legend outside plot
        ax.legend(loc='upper left', bbox_to_anchor=(1.15, 1.0), 
                  framealpha=0.9, edgecolor='gray', fontsize=8)
        
        fig.savefig(figures_dir / 'radar_top5_models.png')
        plt.close(fig)
    
    # =========================================================================
    # 6. HORIZONTAL BAR: All Experiments Ranked
    # =========================================================================
    if len(df) > 0:
        summary = df.groupby('experiment_id').agg({
            'test_accuracy': ['mean', 'std'],
            'ablation_group': 'first'
        }).reset_index()
        summary.columns = ['experiment_id', 'acc_mean', 'acc_std', 'group']
        summary = summary.sort_values('acc_mean', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(summary) * 0.25)))
        
        bar_colors = [COLORS.get(g, 'gray') for g in summary['group']]
        y_pos = range(len(summary))
        
        bars = ax.barh(y_pos, summary['acc_mean'], xerr=summary['acc_std'], 
                       color=bar_colors, capsize=2, height=0.7, alpha=0.85)
        
        ax.set_yticks(y_pos)
        labels = [e.replace('arch_', '').replace('_', ' ')[:20] for e in summary['experiment_id']]
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Test Accuracy (%)')
        ax.set_title('All Experiments: Accuracy Ranking', fontweight='bold')
        
        # Best line
        best_acc = summary['acc_mean'].max()
        ax.axvline(x=best_acc, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(best_acc + 0.3, len(summary) - 1, f'Best: {best_acc:.1f}%', 
                color='red', fontsize=9, va='center')
        
        # Legend for groups
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COLORS[g], label=g.replace('_', ' ').title()) 
                          for g in ['architecture', 'batch_size', 'resolution'] if g in COLORS]
        ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
        
        plt.tight_layout()
        fig.savefig(figures_dir / 'bar_all_experiments.png')
        plt.close(fig)
    
    # Reset style
    plt.style.use('default')
    print("✓ Journal-quality plots saved (6 combined figures)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Full factorial ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test (2 epochs)')
    parser.add_argument('--single-seed', action='store_true', help='Use single seed only')
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick, single_seed=args.single_seed)
