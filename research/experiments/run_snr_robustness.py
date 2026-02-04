#!/usr/bin/env python3
"""
SNR Robustness Testing
======================

Tests model performance under different noise levels (0-30 dB).
Simulates real-world RF signal degradation.
"""

import sys
import time
import numpy as np
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from config import ResearchConfig
from src.training import setup_gpu, compile_model, train_model
from src.models import create_rf_densenet, get_model_metrics
from src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset
from src.visualization import close_all_figures


def add_gaussian_noise(images: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise to images at specified SNR level.
    
    Args:
        images: Input images (0-1 normalized)
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy images clipped to [0, 1]
    """
    # Calculate noise power from SNR
    # SNR = 10 * log10(signal_power / noise_power)
    signal_power = np.mean(images ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise_std = np.sqrt(noise_power)
    
    # Add noise
    noise = np.random.randn(*images.shape) * noise_std
    noisy_images = images + noise
    
    # Clip to valid range
    return np.clip(noisy_images, 0, 1)


def run_snr_robustness(quick_test: bool = False):
    """
    Test model robustness at different SNR levels.
    
    Trains model on clean data, tests on noisy data at various SNR levels.
    """
    print("=" * 60)
    print("SNR ROBUSTNESS TESTING")
    print("=" * 60)
    
    config = ResearchConfig()
    
    if not config.training.enable_snr_testing:
        print("⚠️ SNR testing disabled in config. Skipping.")
        return None
    
    results_dir = config.output.results_dir / "snr_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = 2 if quick_test else config.training.epochs
    snr_levels = config.training.snr_levels_db
    
    # Setup
    print("\n[1/4] Setup...")
    setup_gpu(memory_growth=True, seed=config.training.seeds[0])
    
    # Load data
    print("\n[2/4] Loading data...")
    data_dir = config.data.data_dir
    categories, _ = validate_dataset_directory(data_dir, min_classes=2)
    
    X, Y = load_dataset_numpy(
        data_dir=data_dir,
        categories=categories,
        img_size=config.data.img_size,
        max_images_per_class=config.data.max_images_per_class,
        show_progress=True
    )
    print(f"  Loaded: {len(X)} samples")
    
    splits = split_dataset(
        X, Y, test_size=0.15, val_size=0.15, seed=config.training.seeds[0]
    )
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']
    
    # Calculate class weights for imbalance handling
    from sklearn.utils import class_weight
    class_weights_vals = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights_vals))
    
    num_classes = len(categories)
    input_shape = (config.data.img_size[0], config.data.img_size[1], 3)
    
    # Train on clean data
    print("\n[3/4] Training on CLEAN data...")
    print("-" * 60)
    
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
    
    run_dir = results_dir / "clean_model"
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

    
    # Test on clean data first
    clean_loss, clean_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Clean test accuracy: {clean_acc*100:.2f}%")
    
    # Test at each SNR level
    print(f"\n[4/4] Testing at {len(snr_levels)} SNR levels...")
    print("-" * 60)
    
    snr_results = []
    
    for snr in snr_levels:
        # Add noise to test set
        X_test_noisy = add_gaussian_noise(X_test, snr_db=snr)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test_noisy, y_test, verbose=0)
        acc_drop = (clean_acc - test_acc) * 100
        
        snr_results.append({
            'snr_db': snr,
            'test_accuracy': test_acc * 100,
            'accuracy_drop': acc_drop,
            'clean_accuracy': clean_acc * 100
        })
        
        print(f"  SNR {snr:2d} dB: Accuracy = {test_acc*100:.2f}% (drop: {acc_drop:.2f}%)")
    
    # Save results
    results_df = pd.DataFrame(snr_results)
    results_df.to_csv(results_dir / "snr_results.csv", index=False)
    
    # Generate plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df['snr_db'], results_df['test_accuracy'], 'o-', linewidth=2, markersize=8)
    ax.axhline(y=clean_acc*100, color='r', linestyle='--', label='Clean accuracy')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Robustness vs SNR Level', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(results_dir / 'snr_robustness.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print("\n" + "=" * 60)
    print(f"SNR ROBUSTNESS COMPLETE")
    print(f"Results saved to: {results_dir}")
    print("=" * 60)
    
    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    run_snr_robustness(quick_test=args.quick)
