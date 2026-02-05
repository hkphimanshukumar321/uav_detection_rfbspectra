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

    # Process runs
    processed_count = 0
    max_plots = 3
    
    count_dict = {
        'architecture': 0,
        'batch_size': 0,
        'resolution': 0
    }
    
    # Heuristic: Find representative runs
    print("Scanning for representative runs...")
    
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir(): continue
        
        model_file = run_path / "best_model.h5"
        if not model_file.exists(): continue
        
        # Categorize run
        name = run_path.name
        group = 'other'
        if 'arch_' in name: group = 'architecture'
        elif 'batch_' in name: group = 'batch_size'
        elif 'res_' in name: group = 'resolution'
        
        # Only plot one per group to avoid clutter, or specifically high-res ones
        target_group_limit = 1
        if count_dict.get(group, 0) >= target_group_limit:
            continue
            
        print(f"Generating CM for: {name} (Group: {group})")
        
        try:
            model = tf.keras.models.load_model(str(model_file))
            
            # Inference
            y_pred_probs = model.predict(X_test, verbose=1, batch_size=32)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            # Compute CM
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot using project utility
            plot_confusion_matrix(
                confusion_matrix=cm,
                class_labels=categories,
                title=f"Confusion Matrix\n{name}",
                normalize=True,
                save_path=figures_dir / f"confusion_matrix_{name}",
                figsize=(12, 10)
            )
            
            print(f"Saved figure for {name}")
            count_dict[group] = count_dict.get(group, 0) + 1
            processed_count += 1
            
            if processed_count >= 5: # Total safety limit
                break
                
        except Exception as e:
            print(f"Failed to process {name}: {e}")

    print("Done. Please git add/commit/push the new figures.")
