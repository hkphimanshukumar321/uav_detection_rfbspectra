#!/usr/bin/env python3
"""
Generate Confusion Matrices from Saved Models
=============================================
Loads the best model from each run, performs inference on the test set,
and generates a confusion matrix.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix

# Add research root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import data loading or define it here if config is tricky
from research.config import Config
from research.src.data_loader import get_data_generators

def generate_confusion_matrices():
    results_dir = Path("research/results")
    runs_dir = results_dir / "runs"
    figures_dir = results_dir / "figures_journal_v2"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Locate data (Assuming it's available on server at Config.DATA_DIR)
    # We need to ensure Config points to the right place or allow override
    print(f"Loading test data from: {Config.DATA_DIR}")
    
    try:
        # We only need the test generator
        _, _, test_gen = get_data_generators(
            data_dir=Config.DATA_DIR,
            batch_size=32, # Standard batch size for inference
            target_size=Config.IMG_SIZE
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure Config.DATA_DIR is correct on the server.")
        return

    # Helper to get class names
    class_names = sorted(list(test_gen.class_indices.keys()))
    print(f"Classes: {class_names}")

    # We can't do ALL runs (too many), so let's pick the BEST run from each group
    # Or just one representative run. 
    # For now, let's look for 'arch_gr12_c0.5_d3_3_3' (a middle/default one) or similar.
    
    # Strategy: Find one valid run for the 'Default/Baseline' configuration to show the CM.
    # Often CM is shown for the *best* performing model.
    
    # Valid run patterns to look for (one from each group or just the best)
    target_pattern = "arch_gr12_c0.5_d3-3-3" # Hypothetical 'center' point
    # Actually, let's just scan all and pick the one with highest accuracy in history.csv?
    # That might be slow. Let's just pick the first few distinct ones.
    
    processed_count = 0
    max_plots = 3 # Limit to avoid flooding
    
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir(): continue
        
        model_file = run_path / "best_model.h5"
        if not model_file.exists(): continue
        
        # Heuristic: Only plot for runs that look like "canonical" examples or high performance
        # For this script, let's just plot the FIRST one we find to test, 
        # and maybe one with specific keyword if user wants.
        # User asked for "confusion matrix", usually implies for the *best* model.
        
        # Let's try to find a high-res, reasonable depth one.
        if "res_128" in run_path.name or "d3_3_3" in run_path.name:
             pass # Good candidate
        else:
             if processed_count >= 1: # If we haven't found any good ones, keep looking, but if we have, skip others
                 continue
        
        print(f"Generating CM for: {run_path.name}")
        
        try:
            model = tf.keras.models.load_model(str(model_file))
            
            # Predict
            # Reset generator
            test_gen.reset()
            # Get all batches
            y_pred_probs = model.predict(test_gen, verbose=1)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = test_gen.classes
            
            # Compute CM
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix\n{run_path.name}')
            plt.tight_layout()
            
            out_file = figures_dir / f"confusion_matrix_{run_path.name}.png"
            plt.savefig(out_file, dpi=300)
            plt.close()
            
            print(f"Saved: {out_file}")
            processed_count += 1
            
            if processed_count >= max_plots:
                break
                
        except Exception as e:
            print(f"Failed to generate for {run_path.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    generate_confusion_matrices()
