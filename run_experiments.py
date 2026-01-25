# -*- coding: utf-8 -*-
"""
Main Experiment Runner for DroneRFB-Spectra Framework
Usage: python run_experiments.py --mode full
"""

import os, sys, argparse, logging, time, json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent))

from src.config import ExperimentConfig, DEFAULT_CONFIG
from src.data_loader import validate_dataset_directory, load_dataset_numpy, split_dataset, get_class_weights
from src.models import create_rf_densenet, create_baseline_model, create_simple_cnn_baseline, get_model_metrics, BASELINE_MODELS
from src.training import setup_gpu, get_device_info, generate_run_id, train_model, compile_model, compute_metrics, compute_confusion_matrix, benchmark_inference, save_training_results, append_to_aggregate_csv, create_callbacks
from src.visualization import plot_confusion_matrix, plot_training_history, plot_radar_chart, plot_roc_curves, plot_ablation_study, plot_model_comparison_bar, plot_class_distribution, close_all_figures

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner for comprehensive model evaluation."""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.data_dir = Path(self.config.data.data_dir)
        self.runs_dir = Path(self.config.output.runs_dir)
        self.categories = None
        self.device_info = None
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        Path(self.config.output.figures_dir).mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Initialize GPU and validate dataset."""
        logger.info("=" * 60)
        logger.info("DroneRFB-Spectra Experiment Runner")
        setup_gpu(memory_growth=True, mixed_precision=True, seed=self.config.training.seed)
        self.device_info = get_device_info()
        self.categories, class_counts = validate_dataset_directory(self.data_dir)
        logger.info(f"Classes: {len(self.categories)}, Samples: {sum(class_counts.values())}")
        plot_class_distribution(class_counts, save_path=Path(self.config.output.figures_dir) / 'class_distribution')
        close_all_figures()
    
    def load_data(self) -> Dict[str, tuple]:
        X, Y = load_dataset_numpy(self.data_dir, self.categories, img_size=self.config.data.img_size)
        return split_dataset(X, Y, test_size=self.config.data.test_split, val_size=self.config.data.val_split, seed=self.config.training.seed)
    
    def run_experiment(self, splits, model_name="RF_DenseNet", model_params=None, training_params=None, prefix="run"):
        model_params = model_params or {}; training_params = training_params or {}
        X_train, y_train = splits['train']; X_val, y_val = splits['val']; X_test, y_test = splits['test']
        run_id = generate_run_id(prefix); run_dir = self.runs_dir / run_id; run_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name == "RF_DenseNet":
            model = create_rf_densenet(X_train.shape[1:], len(self.categories), **model_params)
        elif model_name in BASELINE_MODELS:
            model = create_baseline_model(model_name, X_train.shape[1:], len(self.categories))
        else:
            model = create_simple_cnn_baseline(X_train.shape[1:], len(self.categories))
        
        model = compile_model(model, learning_rate=training_params.get('learning_rate', 1e-3))
        history = train_model(model, X_train, y_train, X_val, y_val, run_dir, epochs=training_params.get('epochs', 50), batch_size=training_params.get('batch_size', 32), class_weights=get_class_weights(y_train))
        
        y_prob = model.predict(X_test, verbose=0); y_pred = np.argmax(y_prob, axis=1)
        metrics = compute_metrics(y_test, y_pred, y_prob, self.categories)
        cm, _ = compute_confusion_matrix(y_test, y_pred, self.categories)
        latency = benchmark_inference(model, X_train.shape[1:])
        
        save_training_results(run_dir, {'run_id': run_id, 'model': model_name, **model_params}, history, metrics, cm, self.categories, latency, self.device_info)
        plot_confusion_matrix(cm, self.categories, save_path=run_dir / 'confusion_matrix')
        plot_training_history(history, save_path=run_dir / 'training_history')
        close_all_figures()
        
        append_to_aggregate_csv(self.runs_dir / '_aggregate_metrics.csv', {'run_id': run_id, 'model': model_name, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}})
        logger.info(f"{run_id}: Accuracy={metrics['accuracy']:.4f}")
        return {'run_id': run_id, 'metrics': metrics, 'latency': latency}
    
    def run_ablation(self, splits):
        logger.info("Starting Ablation Study...")
        results = []
        for gr in [4, 8, 12, 16]:
            r = self.run_experiment(splits, "RF_DenseNet", {'growth_rate': gr, 'compression': 0.5, 'depth': (3,3,3)}, prefix=f"abl_gr{gr}")
            results.append({'type': 'growth_rate', 'value': gr, **r['metrics']})
        for c in [0.25, 0.5, 0.75]:
            r = self.run_experiment(splits, "RF_DenseNet", {'growth_rate': 8, 'compression': c, 'depth': (3,3,3)}, prefix=f"abl_c{c}")
            results.append({'type': 'compression', 'value': c, **r['metrics']})
        for d in [(2,2,2), (3,3,3), (4,4,4)]:
            r = self.run_experiment(splits, "RF_DenseNet", {'growth_rate': 8, 'compression': 0.5, 'depth': d}, prefix=f"abl_d{d}")
            results.append({'type': 'depth', 'value': str(d), **r['metrics']})
        pd.DataFrame(results).to_csv(self.runs_dir / '_ablation.csv', index=False)
        return pd.DataFrame(results)
    
    def run_baselines(self, splits):
        logger.info("Starting Baseline Comparison...")
        results = []
        r = self.run_experiment(splits, "RF_DenseNet", {'growth_rate': 8, 'compression': 0.5, 'depth': (3,3,3)}, prefix="ours")
        results.append({'model': 'RF-DenseNet (Ours)', **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)}})
        for m in ['MobileNetV2', 'DenseNet121', 'VGG16', 'ResNet50V2']:
            try:
                r = self.run_experiment(splits, m, prefix=f"base_{m.lower()}")
                results.append({'model': m, **{k: v for k, v in r['metrics'].items() if not isinstance(v, dict)}})
            except Exception as e:
                logger.error(f"Failed {m}: {e}")
        df = pd.DataFrame(results)
        df.to_csv(self.runs_dir / '_comparison.csv', index=False)
        plot_model_comparison_bar(df, 'accuracy', 'model', save_path=Path(self.config.output.figures_dir) / 'comparison')
        close_all_figures()
        return df
    
    def run_full(self):
        self.setup(); splits = self.load_data()
        self.run_ablation(splits); self.run_baselines(splits)
        logger.info(f"Complete! Results in {self.runs_dir}")

def main():
    parser = argparse.ArgumentParser(description="DroneRFB-Spectra Experiments")
    parser.add_argument('--mode', choices=['full', 'ablation', 'baseline'], default='full')
    parser.add_argument('--data-dir', default='nr1 1 (1)')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.data.data_dir = Path(args.data_dir)
    config.training.epochs = args.epochs
    runner = ExperimentRunner(config)
    
    if args.mode == 'full': runner.run_full()
    elif args.mode == 'ablation': runner.setup(); runner.run_ablation(runner.load_data())
    elif args.mode == 'baseline': runner.setup(); runner.run_baselines(runner.load_data())

if __name__ == "__main__":
    main()
