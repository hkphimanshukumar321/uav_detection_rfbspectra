#!/usr/bin/env python3
"""
Lightweight Confusion Matrix Generator
=======================================
Generates ONE confusion matrix from the best performing model.
Uses only 500 samples to avoid memory issues.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from research.config import ResearchConfig
from research.src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset
from research.src.visualization import plot_confusion_matrix

def main():
    results_dir = Path("research/results")
    runs_dir = results_dir / "runs"
    figures_dir = results_dir / "figures_journal_v2"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = ResearchConfig()
    
    # Find data directory
    print("Locating data directory...")
    data_dir = config.data.data_dir
    possible_paths = [
        Path("data"), 
        Path("/home/himanshuk/DRONE_RFB_SPECTRA/uav_detection_rfbspectra/data"),
        data_dir,
    ]
    
    final_data_path = None
    for p in possible_paths:
        if p.exists() and p.is_dir():
            final_data_path = p
            print(f"Found data at: {final_data_path}")
            break
            
    if final_data_path is None:
        print(f"ERROR: Could not find data directory.")
        return
    
    # Validate dataset
    print("Validating dataset...")
    categories, _ = validate_dataset_directory(final_data_path)
    print(f"Classes: {len(categories)} total")
    
    # Load ONLY 500 samples (much faster, less memory)
    print("Loading sample of dataset (500 images max)...")
    X, Y = load_dataset_numpy(
        data_dir=final_data_path,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=20,  # 20 per class × 25 classes = 500 total
        show_progress=True
    )
    
    # Split
    print("Splitting dataset...")
    splits = split_dataset(X, Y, test_size=0.3, val_size=0.2, seed=42)
    X_test, y_test = splits['test']
    print(f"Test set: {len(X_test)} samples")
    
    # Find the first valid model
    print("Scanning for models...")
    model_path = None
    run_name = None
    
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir(): continue
        model_file = run_path / "best_model.h5"
        if model_file.exists():
            model_path = model_file
            run_name = run_path.name
            print(f"Found model: {run_name}")
            break
    
    if model_path is None:
        print("ERROR: No models found in research/results/runs/")
        return
    
    # Load model and predict
    print("Loading model...")
    model = tf.keras.models.load_model(str(model_path))
    
    print("Running inference...")
    y_pred_probs = model.predict(X_test, verbose=0, batch_size=32)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Compute confusion matrix
    print("Computing confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    print("Generating plot...")
    plot_confusion_matrix(
        confusion_matrix=cm,
        class_labels=categories,
        title=f"Confusion Matrix (Sample)\n{run_name}",
        normalize=True,
        save_path=figures_dir / f"confusion_matrix_sample",
        figsize=(12, 10)
    )
    
    print(f"✓ Saved: {figures_dir / 'confusion_matrix_sample.png'}")
    print("Done!")

if __name__ == "__main__":
    main()
