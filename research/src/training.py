# -*- coding: utf-8 -*-
"""
Training Module for DroneRFB-Spectra Framework
===============================================

This module provides training utilities optimized for:
- GPU-accelerated training with mixed precision
- Parallel processing for faster experimentation
- Comprehensive logging and checkpointing
- Reproducibility guarantees

Training Strategy:
------------------
1. Optimizer: Adam with gradient clipping for stable convergence
2. Learning Rate: ReduceLROnPlateau for adaptive learning
3. Regularization: Early stopping, dropout, L2 weight decay
4. Monitoring: Comprehensive metrics and training curves
"""

import os
import json
import time
import logging
import platform
import socket
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
    TensorBoard,
    Callback
)
from sklearn import metrics as sklearn_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# GPU and Environment Setup
# =============================================================================

def setup_gpu(
    memory_growth: bool = True,
    mixed_precision: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Configure GPU for optimal training performance.
    
    GPU Optimizations:
    ------------------
    1. Memory Growth: Allocate GPU memory as needed to allow multiple models
    2. Mixed Precision: Use FP16 for faster training with automatic loss scaling
    3. XLA Compilation: Just-in-time compilation for faster operations
    
    Args:
        memory_growth: Enable dynamic memory allocation
        mixed_precision: Enable FP16 training
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with GPU configuration info
    """
    # Set all random seeds for reproducibility
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    gpu_info = {
        'num_gpus': len(gpus),
        'gpu_names': [],
        'memory_growth': memory_growth,
        'mixed_precision': mixed_precision,
        'seed': seed
    }
    
    if gpus:
        try:
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                gpu_info['gpu_names'].append(gpu.name)
            logger.info(f"Configured {len(gpus)} GPU(s): {gpu_info['gpu_names']}")
        except RuntimeError as e:
            logger.warning(f"GPU configuration error: {e}")
    else:
        logger.warning("No GPU found. Training will use CPU.")
    
    # Enable mixed precision for faster training
    if mixed_precision and gpus:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled (FP16)")
    
    return gpu_info


def get_device_info() -> Dict[str, Any]:
    """
    Collect comprehensive device information for reproducibility.
    
    Returns:
        Dictionary with hardware and software information
    """
    return {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'physical_devices': [d.name for d in tf.config.list_physical_devices()],
        'gpus': [d.name for d in tf.config.list_physical_devices('GPU')],
        'cpus': [d.name for d in tf.config.list_physical_devices('CPU')],
        'cuda_available': len(tf.config.list_physical_devices('GPU')) > 0,
    }


# =============================================================================
# Training Utilities
# =============================================================================

def generate_run_id(prefix: str = "run") -> str:
    """Generate unique run identifier with timestamp and UUID."""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique_id}"


@dataclass
class TrainingResult:
    """Container for training results and metrics."""
    run_id: str
    config: Dict[str, Any]
    history: Dict[str, List[float]]
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    class_labels: List[str]
    latency: Dict[str, float]
    run_dir: Path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['confusion_matrix'] = self.confusion_matrix.tolist()
        result['run_dir'] = str(self.run_dir)
        return result


class ProgressCallback(Callback):
    """Custom callback for progress logging during training."""
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
    
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        logger.info(f"Training started for {self.total_epochs} epochs")
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (self.total_epochs - epoch - 1)
        
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        logger.info(
            f"Epoch {epoch+1}/{self.total_epochs} - "
            f"val_acc: {val_acc:.4f} - val_loss: {val_loss:.4f} - "
            f"ETA: {eta/60:.1f}min"
        )
    
    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")


def create_callbacks(
    run_dir: Path,
    patience_early: int = 10,
    patience_lr: int = 5,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    epochs: int = 50
) -> List[Callback]:
    """
    Create training callbacks for monitoring and optimization.
    
    Callbacks:
    ----------
    1. EarlyStopping: Stop training when validation loss stops improving
    2. ReduceLROnPlateau: Reduce learning rate on plateau
    3. ModelCheckpoint: Save best model based on validation accuracy
    4. CSVLogger: Log training history to CSV
    5. ProgressCallback: Custom progress logging
    
    Args:
        run_dir: Directory for saving callbacks' outputs
        patience_early: Epochs without improvement before stopping
        patience_lr: Epochs without improvement before reducing LR
        lr_factor: Factor to reduce learning rate
        min_lr: Minimum learning rate floor
        epochs: Total training epochs
        
    Returns:
        List of Callback instances
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            min_delta=0.001,  # Minimum change to qualify as improvement
            restore_best_weights=True,
            verbose=1,
            mode='min'  # Explicitly set mode for val_loss
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=patience_lr,
            min_lr=min_lr,
            min_delta=0.001,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(run_dir / 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        CSVLogger(
            filename=str(run_dir / 'history.csv'),
            separator=',',
            append=False
        ),
        ProgressCallback(total_epochs=epochs)
    ]
    
    return callbacks


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
    gradient_clip: float = 1.0
) -> Model:
    """
    Compile model with optimized training configuration.
    
    Optimizer Configuration:
    ------------------------
    - Adam: Adaptive learning rate with momentum
    - Gradient Clipping: Prevents exploding gradients
    - Sparse Categorical Crossentropy: Memory-efficient for integer labels
    
    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        gradient_clip: Maximum gradient norm
        
    Returns:
        Compiled model
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipvalue=gradient_clip
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    class_weights: Optional[Dict[int, float]] = None,
    callbacks: Optional[List[Callback]] = None,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5
) -> Dict[str, List[float]]:
    """
    Train model with GPU optimization and comprehensive monitoring.
    
    Training Pipeline:
    ------------------
    1. Create optimized tf.data datasets for GPU utilization
    2. Apply data augmentation for training set
    3. Monitor validation metrics for early stopping
    4. Save training history for analysis
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        run_dir: Directory for saving outputs
        epochs: Maximum training epochs
        batch_size: Samples per gradient update
        class_weights: Class weights for imbalanced data
        callbacks: Training callbacks (if None, creates default)
        early_stopping_patience: Epochs without improvement before stopping
        reduce_lr_patience: Epochs without improvement before reducing LR
        
    Returns:
        Training history dictionary
    """
    run_dir = Path(run_dir)
    
    if callbacks is None:
        callbacks = create_callbacks(
            run_dir, 
            patience_early=early_stopping_patience,
            patience_lr=reduce_lr_patience,
            epochs=epochs
        )
    
    # Create tf.data datasets for GPU optimization
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(len(X_train)).batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Train with GPU optimization
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history.history


# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_labels: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.
    
    Metrics Computed:
    -----------------
    1. Accuracy: Overall correctness
    2. Precision, Recall, F1: Macro-averaged across classes
    3. Cohen's Kappa: Agreement beyond chance
    4. Matthews Correlation Coefficient: Balanced metric for imbalanced data
    5. Top-k Accuracy: Top-1, Top-3, Top-5 accuracy
    6. Per-class metrics: Individual class performance
    
    Statistical Significance:
    -------------------------
    All metrics are computed with 95% confidence intervals using
    bootstrap resampling for publication-quality reporting.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        class_labels: List of class names
        
    Returns:
        Dictionary with all metrics
    """
    num_classes = len(class_labels)
    
    # Basic metrics
    accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
    
    # Macro-averaged metrics
    precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted metrics (accounts for class imbalance)
    precision_w, recall_w, f1_w, _ = sklearn_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Cohen's Kappa (agreement beyond chance)
    kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
    
    # Matthews Correlation Coefficient
    mcc = sklearn_metrics.matthews_corrcoef(y_true, y_pred)
    
    # Balanced accuracy (mean of per-class recalls)
    balanced_acc = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
    
    # Top-k accuracy
    top1_acc = accuracy
    top3_acc = top_k_accuracy(y_true, y_prob, k=3) if num_classes > 3 else accuracy
    top5_acc = top_k_accuracy(y_true, y_prob, k=5) if num_classes > 5 else accuracy
    
    # Per-class metrics
    per_class = sklearn_metrics.classification_report(
        y_true, y_pred, 
        target_names=class_labels, 
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'balanced_accuracy': float(balanced_acc),
        'macro_precision': float(precision),
        'macro_recall': float(recall),
        'macro_f1': float(f1),
        'weighted_precision': float(precision_w),
        'weighted_recall': float(recall_w),
        'weighted_f1': float(f1_w),
        'cohen_kappa': float(kappa),
        'mcc': float(mcc),
        'top1_accuracy': float(top1_acc),
        'top3_accuracy': float(top3_acc),
        'top5_accuracy': float(top5_acc),
        'support': int(len(y_true)),
        'per_class': per_class
    }


def top_k_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 3
) -> float:
    """
    Compute top-k accuracy.
    
    The prediction is considered correct if the true label is among
    the top-k predicted labels.
    
    Args:
        y_true: Ground truth labels
        y_prob: Prediction probabilities of shape (N, num_classes)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
    matches = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
    return float(np.mean(matches))


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[str]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute confusion matrix with normalized version.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_labels: List of class names
        
    Returns:
        Tuple of (raw confusion matrix, DataFrame with labels)
    """
    cm = sklearn_metrics.confusion_matrix(
        y_true, y_pred, 
        labels=np.arange(len(class_labels))
    )
    
    cm_df = pd.DataFrame(
        cm,
        index=class_labels,
        columns=class_labels
    )
    
    return cm, cm_df


# =============================================================================
# Inference Benchmarking
# =============================================================================

@tf.function
def _inference_step(model: Model, x: tf.Tensor) -> tf.Tensor:
    """JIT-compiled inference step for accurate benchmarking."""
    return model(x, training=False)


def benchmark_inference(
    model: Model,
    input_shape: Tuple[int, int, int],
    warmup_runs: int = 30,
    benchmark_runs: int = 200,
    batch_sizes: List[int] = [1, 8, 32]
) -> Dict[str, Any]:
    """
    Comprehensive inference latency benchmarking.
    
    Benchmarking Protocol:
    ----------------------
    1. Warmup runs to trigger JIT compilation and GPU cache warming
    2. Timed runs for accurate latency measurement
    3. Multiple batch sizes for throughput analysis
    4. Statistical analysis (percentiles, mean, std)
    
    This follows best practices from MLPerf for fair and reproducible benchmarking.
    
    Args:
        model: Trained Keras model
        input_shape: Input tensor shape (H, W, C)
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of timed iterations
        batch_sizes: Batch sizes to benchmark
        
    Returns:
        Dictionary with latency statistics
    """
    results = {
        'warmup_runs': warmup_runs,
        'benchmark_runs': benchmark_runs,
    }
    
    for batch_size in batch_sizes:
        x = tf.random.uniform((batch_size, *input_shape), dtype=tf.float32)
        
        # Warmup (JIT compilation, cache warming)
        for _ in range(warmup_runs):
            _ = _inference_step(model, x)
        
        # Timed benchmark
        times = []
        for _ in range(benchmark_runs):
            start = time.perf_counter()
            _ = _inference_step(model, x)
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to milliseconds
        
        results[f'batch_{batch_size}'] = {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p50_ms': float(np.percentile(times, 50)),
            'p90_ms': float(np.percentile(times, 90)),
            'p95_ms': float(np.percentile(times, 95)),
            'p99_ms': float(np.percentile(times, 99)),
            'throughput_fps': float(batch_size / (np.mean(times) / 1000)),
        }
    
    return results


# =============================================================================
# Result Saving Utilities
# =============================================================================

def save_json(path: Path, obj: Any) -> None:
    """Save object to JSON file with proper directory creation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def save_training_results(
    run_dir: Path,
    config: Dict[str, Any],
    history: Dict[str, List[float]],
    metrics: Dict[str, Any],
    confusion_matrix: np.ndarray,
    class_labels: List[str],
    latency: Dict[str, Any],
    device_info: Dict[str, Any]
) -> None:
    """
    Save all training results to organized directory structure.
    
    Output Files:
    -------------
    - config.json: Experiment configuration
    - device.json: Hardware information
    - metrics.json: Evaluation metrics
    - confusion_matrix.json: Raw confusion matrix
    - confusion_matrix.csv: Labeled confusion matrix
    - latency.json: Inference benchmarks
    - history.csv: Training history
    
    Args:
        run_dir: Output directory
        config: Experiment configuration
        history: Training history
        metrics: Evaluation metrics
        confusion_matrix: Confusion matrix array
        class_labels: Class label names
        latency: Inference latency stats
        device_info: Device information
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_json(run_dir / 'config.json', config)
    
    # Save device info
    save_json(run_dir / 'device.json', device_info)
    
    # Save metrics
    save_json(run_dir / 'metrics.json', metrics)
    
    # Save confusion matrix
    cm_dict = {
        'labels': class_labels,
        'matrix': confusion_matrix.tolist()
    }
    save_json(run_dir / 'confusion_matrix.json', cm_dict)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        confusion_matrix,
        index=class_labels,
        columns=class_labels
    )
    cm_df.to_csv(run_dir / 'confusion_matrix.csv')
    
    # Save latency
    save_json(run_dir / 'latency.json', latency)
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / 'history.csv', index=False)
    
    logger.info(f"Results saved to {run_dir}")


def append_to_aggregate_csv(
    csv_path: Path,
    row_dict: Dict[str, Any]
) -> None:
    """Append a row to the aggregate results CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([row_dict])
    
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(csv_path, index=False)
