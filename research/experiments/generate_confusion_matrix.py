#!/usr/bin/env python3
"""
Generate Confusion Matrices from Saved Models
=============================================
Loads the best model from each run, performs inference on the test set,
and generates a confusion matrix using the project's visualization tools.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Correct Imports based on codebase verification
from research.config import ResearchConfig
from research.src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset
from research.src.visualization import plot_confusion_matrix, close_all_figures

def generate_confusion_matrices():
    results_dir = Path("research/results")
    runs_dir = results_dir / "runs"
    figures_dir = results_dir / "figures_journal_v2"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Instantiate Config
    config = ResearchConfig()
    
    # Robust data path checking
    data_dir = config.data.data_dir
    possible_paths = [
        data_dir,
        Path("data"), 
        Path("../data"),
        Path("../../data"),
        Path("/home/himanshuk/DRONE_RFB_SPECTRA/uav_detection_rfbspectra/data"),
        Path.home() / "DRONE_RFB_SPECTRA/uav_detection_rfbspectra/data"
    ]
    
    final_data_path = None
    for p in possible_paths:
        if p.exists() and p.is_dir():
            final_data_path = p
            break
            
    if final_data_path is None:
        print(f"Error: Could not find data directory. Checked: {possible_paths}")
        return
        
    print(f"Loading test data from: {final_data_path}")
    
    try:
        # 1. Validate and get categories
        categories, _ = validate_dataset_directory(final_data_path)
        print(f"Classes: {categories}")
        
        # 2. Load Dataset into Memory (as per project pattern)
        # Note: This loads all data, might be heavy but is how the project works
        X, Y = load_dataset_numpy(
            data_dir=final_data_path,
            categories=categories,
            img_size=config.data.img_size,
            show_progress=True
        )
        
        # 3. Split to get Test set
        splits = split_dataset(
            X, Y, 
            test_size=config.data.test_split,
            val_size=config.data.val_split,
            seed=42 # Use fixed seed to match training split
        )
        X_test, y_test = splits['test']
        print(f"Test Set: {len(X_test)} samples")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Target specific best model
    target_run = "arch_gr12_c0.75_d4_4_4_seed42"
    print(f"Targeting best model: {target_run}")
    
    run_path = runs_dir / target_run
    if not run_path.exists():
        print(f"Error: Run directory not found: {run_path}")
        return
        
    model_file = run_path / "best_model.h5"
    if not model_file.exists():
        print(f"Error: Model file not found: {model_file}")
        return
        
    try:
        print(f"Loading model from {model_file}...")
        model = tf.keras.models.load_model(str(model_file))
        
        print("Running inference on test set (full dataset)...")
        y_pred_probs = model.predict(X_test, verbose=1, batch_size=32)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Compute CM
        print("Computing confusion matrix...")
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot using project utility
        print("Generating visualization...")
        save_path = figures_dir / f"confusion_matrix_BEST_{target_run}"
        
        plot_confusion_matrix(
            confusion_matrix=cm,
            class_labels=categories,
            title=f"Confusion Matrix (Best Model)\n{target_run}",
            normalize=True,
            save_path=save_path,
            figsize=(14, 12) # Slightly larger for better readability
        )
        
        print(f"Values exported to: {save_path}.png")
        
    except Exception as e:
        print(f"Failed to process {target_run}: {e}")
        import traceback
        traceback.print_exc()

    print("Done. Please git add/commit/push the new figures.")

if __name__ == "__main__":
    generate_confusion_matrices()
