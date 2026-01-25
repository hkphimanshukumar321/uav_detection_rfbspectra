#!/usr/bin/env python3
"""
Ablation Study - Growth Rate, Compression, and Depth
=====================================================

Systematically evaluates the impact of architectural parameters.
"""

import sys
from pathlib import Path
import pandas as pd

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent))

from config import ResearchConfig, print_experiment_summary
from src.training import train_model, setup_gpu, compile_model, compute_metrics, generate_run_id
from src.models import create_rf_densenet, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset

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
        
        # Store results
        growth_results.append({
            'growth_rate': gr,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params
        })
        
        print(f"✓ Growth Rate {gr}: Test Acc = {test_acc*100:.2f}%")
    
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
        
        # Store results
        compression_results.append({
            'compression': comp,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params
        })
        
        print(f"✓ Compression {comp}: Test Acc = {test_acc*100:.2f}%")
    
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
        
        # Store results
        depth_results.append({
            'depth': str(d),
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'val_accuracy': max(history['val_accuracy']) * 100 if 'val_accuracy' in history else 0,
            'total_params': metrics_info.total_params,
            'trainable_params': metrics_info.trainable_params
        })
        
        print(f"✓ Depth {d}: Test Acc = {test_acc*100:.2f}%")
    
    # Save depth results
    depth_df = pd.DataFrame(depth_results)
    depth_csv = results_dir / "ablation_depth.csv"
    depth_df.to_csv(depth_csv, index=False)
    print(f"\n✓ Saved depth results to: {depth_csv}")
    print(depth_df.to_string(index=False))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE!")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"  - {growth_csv.name}")
    print(f"  - {compression_csv.name}")
    print(f"  - {depth_csv.name}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (2 epochs)')
    args = parser.parse_args()
    
    run_ablation(quick_test=args.quick)
