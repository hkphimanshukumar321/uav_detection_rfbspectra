#!/usr/bin/env python3
"""
Ablation Study - Growth Rate, Compression, and Depth
=====================================================

Systematically evaluates the impact of architectural parameters.
"""

import sys
import time
import numpy as np
from pathlib import Path
import pandas as pd

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


class AblationProgress:
    """
    Track ablation study progress with ETA and visual progress bar.
    
    Shows a filling progress bar like:
    [████████████░░░░░░░░░░░░░░░░░░] 42% - ETA: 3h 25m | Completed: 5/12
    """
    
    def __init__(self, total_experiments: int):
        self.total = total_experiments
        self.completed = 0
        self.start_time = time.time()
        self.experiment_times = []
        self.current_category = ""
    
    def set_category(self, category: str):
        """Set current ablation category being run."""
        self.current_category = category
    
    def update(self, experiment_name: str, duration: float = None):
        """Update progress after completing an experiment."""
        self.completed += 1
        if duration:
            self.experiment_times.append(duration)
        self.display_progress_bar()
    
    def display_progress_bar(self, width: int = 40):
        """Print visual progress bar with ETA."""
        if self.total == 0:
            return
        
        percent = self.completed / self.total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        
        # Calculate ETA based on average experiment time
        if self.experiment_times:
            avg_time = sum(self.experiment_times) / len(self.experiment_times)
            remaining = self.total - self.completed
            eta_seconds = avg_time * remaining
            
            if eta_seconds > 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds > 60:
                eta_str = f"{eta_seconds/60:.0f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
        else:
            eta_str = "calculating..."
        
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed/60:.1f}m" if elapsed > 60 else f"{elapsed:.0f}s"
        
        print(f"\r[{bar}] {percent*100:5.1f}% | {self.completed}/{self.total} | "
              f"Elapsed: {elapsed_str} | ETA: {eta_str} | {self.current_category}", end="", flush=True)
        
        if self.completed == self.total:
            print()  # New line when complete
    
    def get_summary(self) -> dict:
        """Get progress summary for reporting."""
        total_time = time.time() - self.start_time
        return {
            'total_experiments': self.total,
            'completed': self.completed,
            'total_time_seconds': total_time,
            'avg_time_per_experiment': sum(self.experiment_times) / len(self.experiment_times) if self.experiment_times else 0
        }

def run_ablation(quick_test=False):
    """
    Run complete ablation study.
    
    Args:
        quick_test: If True, run with minimal epochs for testing
    """
    print("=" * 70)
    print("ABLATION STUDY - ARCHITECTURAL PARAMETERS")
    print("=" * 70)
    
    # Load configuration
    config = ResearchConfig()
    
    # Setup GPU
    print("\n[1/6] Setting up GPU...")
    gpu_info = setup_gpu(
        memory_growth=True,
        mixed_precision=False,  # Disable for stability
        seed=config.training.seed
    )
    print(f"GPU Configuration: {gpu_info['num_gpus']} GPU(s) available")
    
    # Validate and load dataset
    print("\n[2/6] Loading dataset...")
    data_dir = config.data.data_dir
    
    print(f"Data directory: {data_dir}")
    categories, class_to_idx = validate_dataset_directory(
        data_dir,
        min_classes=2,
        min_samples_per_class=10
    )
    print(f"Found {len(categories)} classes: {categories[:5]}..." if len(categories) > 5 else f"Found {len(categories)} classes: {categories}")
    
    # Load all images
    X, Y = load_dataset_numpy(
        data_dir=data_dir,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )
    print(f"Loaded {len(X)} images, shape: {X.shape}")
    
    # Split dataset
    print("\n[3/6] Splitting dataset...")
    splits = split_dataset(
        X, Y,
        test_size=config.data.test_split,
        val_size=config.data.val_split,
        seed=config.training.seed,
        stratify=True
    )
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create results directory
    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(categories)
    input_shape = (config.data.img_size[0], config.data.img_size[1], 3)
    
    # Use fewer epochs for quick test
    epochs = 2 if quick_test else config.training.epochs
    
    # Initialize progress tracker
    n_growth = len(config.ablation.growth_rates)
    n_comp = len(config.ablation.compressions)
    n_depth = len(config.ablation.depths)
    total_experiments = n_growth + n_comp + n_depth
    
    progress = AblationProgress(total_experiments)
    
    print(f"\n📊 Total experiments to run: {total_experiments}")
    print(f"   - Growth rates: {n_growth}")
    print(f"   - Compressions: {n_comp}")
    print(f"   - Depths: {n_depth}")
    print(f"   - Epochs per experiment: {epochs}")
    
    ablation_start_time = time.time()
    
    # =========================================================================
    # ABLATION 1: Growth Rate
    # =========================================================================
    print("\n[4/6] Running Growth Rate Ablation...")
    print("=" * 70)
    
    growth_results = []
    for gr in config.ablation.growth_rates:
        print(f"\n--- Testing Growth Rate: {gr} ---")
        
        # Create model
        model = create_rf_densenet(
            input_shape=input_shape,
            num_classes=num_classes,
            growth_rate=gr,
            compression=config.model.compression,
            depth=config.model.depth,
            dropout_rate=config.model.dropout_rate,
            initial_filters=config.model.initial_filters
        )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=config.training.learning_rate,
            gradient_clip=config.training.gradient_clip_value
        )
        
        # Get model info
        metrics_info = get_model_metrics(model)
        print(f"Model: {metrics_info}")
        
        # Create run directory
        run_id = generate_run_id(f"ablation_gr{gr}")
        run_dir = results_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        print(f"Training for {epochs} epochs...")
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=config.training.batch_size
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate and save training history plot
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_training_history(
            history=history,
            title=f"Training History - Growth Rate {gr}",
            save_path=figures_dir / f"training_history_gr{gr}.png"
        )
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix, f1_score
        cm = sk_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            confusion_matrix=cm,
            class_labels=categories,
            title=f"Confusion Matrix - Growth Rate {gr}",
            normalize=True,
            save_path=figures_dir / f"confusion_matrix_gr{gr}.png"
        )
        
        # Generate ROC curves (journal-ready)
        plot_roc_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"ROC Curves - Growth Rate {gr}",
            save_path=figures_dir / f"roc_curves_gr{gr}.png"
        )
        
        # Generate Precision-Recall curves (journal-ready)
        plot_precision_recall_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"Precision-Recall Curves - Growth Rate {gr}",
            save_path=figures_dir / f"pr_curves_gr{gr}.png"
        )
        
        # Compute F1 score for metrics
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Benchmark inference time
        input_shape = (config.data.img_size[0], config.data.img_size[1], 3)
        latency_info = benchmark_inference(model, input_shape, warmup_runs=10, benchmark_runs=50)
        
        # Store results with F1 score
        growth_results.append({
            'growth_rate': gr,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params,
            'inference_ms': latency_info['batch_1']['mean_ms'],
            'throughput_fps': latency_info['batch_1']['throughput_fps']
        })
        
        print(f"✓ Growth Rate {gr}: Test Acc = {test_acc*100:.2f}%, Inference = {latency_info['batch_1']['mean_ms']:.2f}ms")
        close_all_figures()  # Free memory
        
        # Update progress bar
        exp_duration = time.time() - ablation_start_time
        progress.set_category("Growth Rate")
        progress.update(f"GR-{gr}", exp_duration)
    
    # Save growth rate results
    growth_df = pd.DataFrame(growth_results)
    growth_csv = results_dir / "ablation_growth_rate.csv"
    growth_df.to_csv(growth_csv, index=False)
    print(f"\n✓ Saved growth rate results to: {growth_csv}")
    print(growth_df.to_string(index=False))
    
    # =========================================================================
    # ABLATION 2: Compression Factor
    # =========================================================================
    print("\n\n[5/6] Running Compression Factor Ablation...")
    print("=" * 70)
    
    compression_results = []
    for comp in config.ablation.compressions:
        print(f"\n--- Testing Compression: {comp} ---")
        
        # Create model
        model = create_rf_densenet(
            input_shape=input_shape,
            num_classes=num_classes,
            growth_rate=config.model.growth_rate,
            compression=comp,
            depth=config.model.depth,
            dropout_rate=config.model.dropout_rate,
            initial_filters=config.model.initial_filters
        )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=config.training.learning_rate,
            gradient_clip=config.training.gradient_clip_value
        )
        
        # Get model info
        metrics_info = get_model_metrics(model)
        print(f"Model: {metrics_info}")
        
        # Create run directory
        run_id = generate_run_id(f"ablation_comp{comp}")
        run_dir = results_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        print(f"Training for {epochs} epochs...")
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=config.training.batch_size
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate and save training history plot
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        plot_training_history(
            history=history,
            title=f"Training History - Compression {comp}",
            save_path=figures_dir / f"training_history_comp{comp}.png"
        )
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            confusion_matrix=cm,
            class_labels=categories,
            title=f"Confusion Matrix - Compression {comp}",
            normalize=True,
            save_path=figures_dir / f"confusion_matrix_comp{comp}.png"
        )
        
        # Generate ROC curves (journal-ready)
        plot_roc_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"ROC Curves - Compression {comp}",
            save_path=figures_dir / f"roc_curves_comp{comp}.png"
        )
        
        # Generate Precision-Recall curves (journal-ready)
        plot_precision_recall_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"Precision-Recall Curves - Compression {comp}",
            save_path=figures_dir / f"pr_curves_comp{comp}.png"
        )
        
        # Compute F1 score
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Benchmark inference time
        latency_info = benchmark_inference(model, input_shape, warmup_runs=10, benchmark_runs=50)
        
        # Store results with F1 score
        compression_results.append({
            'compression': comp,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params,
            'inference_ms': latency_info['batch_1']['mean_ms'],
            'throughput_fps': latency_info['batch_1']['throughput_fps']
        })
        
        print(f"✓ Compression {comp}: Test Acc = {test_acc*100:.2f}%, Inference = {latency_info['batch_1']['mean_ms']:.2f}ms")
        close_all_figures()  # Free memory
        
        # Update progress bar
        exp_duration = time.time() - ablation_start_time
        progress.set_category("Compression")
        progress.update(f"Comp-{comp}", exp_duration)
    
    # Save compression results
    compression_df = pd.DataFrame(compression_results)
    compression_csv = results_dir / "ablation_compression.csv"
    compression_df.to_csv(compression_csv, index=False)
    print(f"\n✓ Saved compression results to: {compression_csv}")
    print(compression_df.to_string(index=False))
    
    # =========================================================================
    # ABLATION 3: Network Depth
    # =========================================================================
    print("\n\n[6/6] Running Network Depth Ablation...")
    print("=" * 70)
    
    depth_results = []
    for d in config.ablation.depths:
        print(f"\n--- Testing Depth: {d} ---")
        
        # Create model
        model = create_rf_densenet(
            input_shape=input_shape,
            num_classes=num_classes,
            growth_rate=config.model.growth_rate,
            compression=config.model.compression,
            depth=d,
            dropout_rate=config.model.dropout_rate,
            initial_filters=config.model.initial_filters
        )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=config.training.learning_rate,
            gradient_clip=config.training.gradient_clip_value
        )
        
        # Get model info
        metrics_info = get_model_metrics(model)
        print(f"Model: {metrics_info}")
        
        # Create run directory
        run_id = generate_run_id(f"ablation_depth{'_'.join(map(str, d))}")
        run_dir = results_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Train model
        print(f"Training for {epochs} epochs...")
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=config.training.batch_size
        )
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions for confusion matrix
        y_pred_prob = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Generate and save training history plot
        figures_dir = results_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        depth_str = '_'.join(map(str, d))
        plot_training_history(
            history=history,
            title=f"Training History - Depth {d}",
            save_path=figures_dir / f"training_history_depth{depth_str}.png"
        )
        
        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            confusion_matrix=cm,
            class_labels=categories,
            title=f"Confusion Matrix - Depth {d}",
            normalize=True,
            save_path=figures_dir / f"confusion_matrix_depth{depth_str}.png"
        )
        
        # Generate ROC curves (journal-ready)
        plot_roc_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"ROC Curves - Depth {d}",
            save_path=figures_dir / f"roc_curves_depth{depth_str}.png"
        )
        
        # Generate Precision-Recall curves (journal-ready)
        plot_precision_recall_curves(
            y_true=y_test,
            y_prob=y_pred_prob,
            class_labels=categories,
            title=f"Precision-Recall Curves - Depth {d}",
            save_path=figures_dir / f"pr_curves_depth{depth_str}.png"
        )
        
        # Compute F1 score
        macro_f1 = f1_score(y_test, y_pred, average='macro') * 100
        
        # Benchmark inference time
        latency_info = benchmark_inference(model, input_shape, warmup_runs=10, benchmark_runs=50)
        
        # Store results with F1 score
        depth_results.append({
            'depth': str(d),
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'macro_f1': macro_f1,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params,
            'inference_ms': latency_info['batch_1']['mean_ms'],
            'throughput_fps': latency_info['batch_1']['throughput_fps']
        })
        
        print(f"✓ Depth {d}: Test Acc = {test_acc*100:.2f}%, Inference = {latency_info['batch_1']['mean_ms']:.2f}ms")
        close_all_figures()  # Free memory
        
        # Update progress bar
        exp_duration = time.time() - ablation_start_time
        progress.set_category("Depth")
        progress.update(f"Depth-{d}", exp_duration)
    
    # Save depth results
    depth_df = pd.DataFrame(depth_results)
    depth_csv = results_dir / "ablation_depth.csv"
    depth_df.to_csv(depth_csv, index=False)
    print(f"\n✓ Saved depth results to: {depth_csv}")
    print(depth_df.to_string(index=False))
    
    # =========================================================================
    # Summary Plots - Comparing all ablation results
    # =========================================================================
    print("\n[7/7] Generating Summary Plots...")
    print("=" * 70)
    
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Growth Rate Ablation Plot
    if len(growth_df) > 0:
        plot_ablation_study(
            ablation_df=growth_df,
            x_col='growth_rate',
            y_col='test_accuracy',
            title="Ablation: Growth Rate vs Test Accuracy",
            xlabel="Growth Rate (k)",
            ylabel="Test Accuracy (%)",
            save_path=figures_dir / "ablation_summary_growth_rate.png"
        )
        print("✓ Generated growth rate ablation plot")
    
    # Compression Ablation Plot
    if len(compression_df) > 0:
        plot_ablation_study(
            ablation_df=compression_df,
            x_col='compression',
            y_col='test_accuracy',
            title="Ablation: Compression Factor vs Test Accuracy",
            xlabel="Compression Factor (θ)",
            ylabel="Test Accuracy (%)",
            save_path=figures_dir / "ablation_summary_compression.png"
        )
        print("✓ Generated compression ablation plot")
    
    # Model Comparison Bar Chart (all experiments combined)
    all_results = []
    for _, row in growth_df.iterrows():
        all_results.append({'model': f"GR-{row['growth_rate']}", 'accuracy': row['test_accuracy'] / 100})
    for _, row in compression_df.iterrows():
        all_results.append({'model': f"Comp-{row['compression']}", 'accuracy': row['test_accuracy'] / 100})
    for _, row in depth_df.iterrows():
        all_results.append({'model': f"Depth-{row['depth']}", 'accuracy': row['test_accuracy'] / 100})
    
    if all_results:
        comparison_df = pd.DataFrame(all_results)
        plot_model_comparison_bar(
            comparison_df=comparison_df,
            metric_col='accuracy',
            model_col='model',
            title="Ablation Study: Model Comparison",
            save_path=figures_dir / "ablation_model_comparison.png"
        )
        print("✓ Generated model comparison bar chart")
    
    # Radar Chart - Multi-metric comparison (journal-ready)
    radar_metrics = {}
    for _, row in growth_df.iterrows():
        model_name = f"GR-{row['growth_rate']}"
        radar_metrics[model_name] = {
            'Accuracy': row['test_accuracy'] / 100,
            'F1 Score': row.get('macro_f1', 0) / 100,
            'Efficiency': 1 - (row['inference_ms'] / max(growth_df['inference_ms'])),  # Inverse of latency
            'Compactness': 1 - (row['total_params'] / max(growth_df['total_params'])),  # Inverse of params
        }
    
    if radar_metrics:
        plot_radar_chart(
            metrics_dict=radar_metrics,
            title="Multi-Metric Model Comparison (Growth Rate Ablation)",
            save_path=figures_dir / "radar_chart_growth_rate.png"
        )
        print("✓ Generated radar chart comparison")
    
    # Accuracy vs Latency plot (journal-ready - Pareto frontier)
    all_efficiency_data = []
    for _, row in growth_df.iterrows():
        all_efficiency_data.append({
            'model': f"GR-{row['growth_rate']}",
            'accuracy': row['test_accuracy'] / 100,
            'avg_latency_ms': row['inference_ms'],
            'total_params': row['total_params']
        })
    for _, row in compression_df.iterrows():
        all_efficiency_data.append({
            'model': f"Comp-{row['compression']}",
            'accuracy': row['test_accuracy'] / 100,
            'avg_latency_ms': row['inference_ms'],
            'total_params': row['total_params']
        })
    for _, row in depth_df.iterrows():
        all_efficiency_data.append({
            'model': f"Depth-{row['depth']}",
            'accuracy': row['test_accuracy'] / 100,
            'avg_latency_ms': row['inference_ms'],
            'total_params': row['total_params']
        })
    
    if all_efficiency_data:
        efficiency_df = pd.DataFrame(all_efficiency_data)
        plot_accuracy_vs_latency(
            comparison_df=efficiency_df,
            accuracy_col='accuracy',
            latency_col='avg_latency_ms',
            params_col='total_params',
            model_col='model',
            title="Accuracy vs. Inference Latency (Pareto Analysis)",
            save_path=figures_dir / "accuracy_vs_latency.png"
        )
        print("✓ Generated accuracy vs latency plot")
    
    close_all_figures()
    
    # =========================================================================
    # Final Summary with Device Info
    # =========================================================================
    device_info = get_device_info()
    total_ablation_time = time.time() - ablation_start_time
    
    print("\n" + "=" * 70)
    print("🎉 ABLATION STUDY COMPLETE!")
    print("=" * 70)
    print(f"\n⏱️ Total Duration: {total_ablation_time/60:.1f} minutes")
    print(f"\n📊 Results saved to: {results_dir}")
    print(f"  - {growth_csv.name}")
    print(f"  - {compression_csv.name}")
    print(f"  - {depth_csv.name}")
    print(f"\n📈 Journal-Ready Figures saved to: {figures_dir}")
    print(f"  - Training history plots (accuracy/loss curves)")
    print(f"  - Confusion matrices (normalized)")
    print(f"  - ROC curves (per-class AUC)")
    print(f"  - Precision-Recall curves")
    print(f"  - Ablation summary plots")
    print(f"  - Radar chart (multi-metric comparison)")
    print(f"  - Accuracy vs Latency (Pareto analysis)")
    print(f"\n🖥️ Device Information:")
    print(f"  - Hostname: {device_info.get('hostname', 'N/A')}")
    print(f"  - Platform: {device_info.get('platform', 'N/A')}")
    print(f"  - TensorFlow: {device_info.get('tensorflow_version', 'N/A')}")
    print(f"  - GPUs: {device_info.get('gpus', [])}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (2 epochs)')
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick)
