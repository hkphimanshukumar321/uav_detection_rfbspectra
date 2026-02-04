#!/usr/bin/env python3
"""
5-Fold Cross-Validation Experiment
===================================

Validates model robustness across different data splits.
Uses the best configuration from ablation study.
"""

import sys
import time
import numpy as np
from pathlib import Path
import pandas as pd
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.utils import class_weight

sys.path.append(str(Path(__file__).parent.parent))

from config import ResearchConfig
from src.training import setup_gpu, compile_model, train_model, benchmark_inference
from src.models import create_rf_densenet, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset_numpy
from src.visualization import plot_training_history, close_all_figures


def run_cross_validation(quick_test: bool = False):
    """
    Run 5-fold stratified cross-validation.
    
    Uses default model configuration for CV.
    Returns mean ± std accuracy across folds.
    """
    print("=" * 60)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 60)
    
    config = ResearchConfig()
    
    if not config.training.enable_cross_validation:
        print("⚠️ Cross-validation disabled in config. Skipping.")
        return None
    
    results_dir = config.output.results_dir / "cross_validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 2 if quick_test else config.training.epochs
    n_folds = config.training.cv_folds
    
    # Setup
    print("\n[1/3] Setup...")
    setup_gpu(memory_growth=True, seed=config.training.seeds[0])
    
    # Load data
    print("\n[2/3] Loading data...")
    data_dir = config.data.data_dir
    categories, _ = validate_dataset_directory(data_dir, min_classes=2)
    
    X, Y = load_dataset_numpy(
        data_dir=data_dir,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )
    print(f"  Loaded: {len(X)} samples, {len(categories)} classes")
    
    num_classes = len(categories)
    input_shape = (config.data.img_size[0], config.data.img_size[1], 3)
    
    # Run CV
    print(f"\n[3/3] Running {n_folds}-fold CV...")
    print("-" * 60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.training.seeds[0])
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, Y), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        
        # Validation split from training (stratified)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, stratify=y_train, 
            random_state=config.training.seeds[0]
        )
        
        # Calculate class weights
        class_weights_vals = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = dict(enumerate(class_weights_vals))
        
        # Create model
        model = create_rf_densenet(
            input_shape=input_shape,
            num_classes=num_classes,
            growth_rate=config.model.growth_rate,
            compression=config.model.compression,
            depth=config.model.depth,
            dropout_rate=config.model.dropout_rate,
            initial_filters=config.model.initial_filters
        )
        model = compile_model(model, learning_rate=config.training.learning_rate)
        
        # Train
        run_dir = results_dir / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        history = train_model(
            model=model,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            run_dir=run_dir,
            epochs=epochs,
            batch_size=config.training.batch_size,
            class_weights=class_weights,
            early_stopping_patience=config.training.early_stopping_patience
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        fold_results.append({
            'fold': fold,
            'test_accuracy': test_acc * 100,
            'test_loss': test_loss,
            'val_accuracy': max(history.get('val_accuracy', [0])) * 100
        })
        
        print(f"  Fold {fold}: Accuracy = {test_acc*100:.2f}%")
        close_all_figures()
        tf.keras.backend.clear_session()
    
    # Save results
    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(results_dir / "cv_results.csv", index=False)
    
    mean_acc = results_df['test_accuracy'].mean()
    std_acc = results_df['test_accuracy'].std()
    
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION RESULTS: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_cross_validation(quick_test=args.quick)
