# -*- coding: utf-8 -*-
"""
Data Loading Module for DroneRFB-Spectra Framework
===================================================

This module handles all data loading, preprocessing, and augmentation for
RF spectrogram classification. Designed for:
- Efficient parallel data loading using tf.data
- Memory-efficient processing for large datasets
- Robust error handling for corrupted files
- GPU-optimized data pipelines

Technical Details:
------------------
RF spectrograms capture the electromagnetic signature of drones in the
frequency-time domain. Preprocessing steps are designed to:
1. Preserve spectral characteristics while normalizing intensity
2. Enable efficient batch processing on GPU
3. Apply domain-appropriate augmentations
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrityError(Exception):
    """Exception raised for data integrity issues."""
    pass


def validate_dataset_directory(
    data_dir: Path,
    min_classes: int = 2,
    min_samples_per_class: int = 10
) -> Tuple[List[str], Dict[str, int]]:
    """
    Validate dataset directory structure and return class information.
    
    This function performs comprehensive validation to ensure data integrity:
    1. Verifies directory existence
    2. Identifies valid class subdirectories
    3. Counts samples per class
    4. Checks for minimum class requirements
    
    Args:
        data_dir: Root directory containing class subfolders
        min_classes: Minimum number of classes required
        min_samples_per_class: Minimum samples per class required
        
    Returns:
        Tuple of (class_names, class_counts) where:
        - class_names: Sorted list of class directory names
        - class_counts: Dictionary mapping class names to sample counts
        
    Raises:
        DataIntegrityError: If validation fails
        
    Example:
        >>> classes, counts = validate_dataset_directory(Path("dataset"))
        >>> print(f"Found {len(classes)} classes")
        Found 25 classes
    """
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        raise DataIntegrityError(f"Dataset directory does not exist: {data_dir}")
    
    if not data_dir.is_dir():
        raise DataIntegrityError(f"Path is not a directory: {data_dir}")
    
    # Find valid class directories (numeric names 0-23 in our case)
    class_dirs = [
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    ]
    
    # Sort numerically if possible, otherwise alphabetically
    try:
        class_names = sorted(class_dirs, key=int)
    except ValueError:
        class_names = sorted(class_dirs)
    
    if len(class_names) < min_classes:
        raise DataIntegrityError(
            f"Insufficient classes: found {len(class_names)}, "
            f"minimum required: {min_classes}"
        )
    
    # Count samples per class
    class_counts = {}
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    for class_name in class_names:
        class_path = data_dir / class_name
        count = sum(
            1 for f in class_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        )
        class_counts[class_name] = count
        
        if count < min_samples_per_class:
            logger.warning(
                f"Class '{class_name}' has only {count} samples "
                f"(minimum recommended: {min_samples_per_class})"
            )
    
    total_samples = sum(class_counts.values())
    logger.info(
        f"Dataset validated: {len(class_names)} classes, "
        f"{total_samples} total samples"
    )
    
    return class_names, class_counts


def load_image(
    path: Path,
    img_size: Tuple[int, int] = (64, 64),
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image with error handling.
    
    Preprocessing Pipeline:
    ----------------------
    1. Load image using OpenCV (BGR format)
    2. Convert to RGB color space
    3. Resize using INTER_AREA for downscaling (preserves spectral detail)
    4. Normalize to [0, 1] range (optional)
    
    The INTER_AREA interpolation is specifically chosen for RF spectrograms
    as it provides better anti-aliasing for downscaling compared to bilinear,
    which is crucial for preserving frequency patterns.
    
    Args:
        path: Path to image file
        img_size: Target size (height, width)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Preprocessed image array or None if loading fails
        
    Note:
        Returns None instead of raising exception for corrupted files
        to allow batch processing to continue with warnings.
    """
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f"Failed to load image: {path}")
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize with INTER_AREA for better downscaling quality
        img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
        
        if normalize:
            img = img.astype(np.float32) / 255.0
        
        return img
        
    except Exception as e:
        logger.warning(f"Error loading image {path}: {e}")
        return None


def load_dataset_numpy(
    data_dir: Path,
    categories: List[str],
    img_size: Tuple[int, int] = (64, 64),
    max_images_per_class: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load entire dataset into NumPy arrays.
    
    This function loads all images into memory for training. For large datasets,
    consider using create_tf_dataset() for streaming data loading instead.
    
    Memory Estimation:
    ------------------
    Memory = num_samples × height × width × channels × 4 bytes
    Example: 14,500 × 64 × 64 × 3 × 4 = ~710 MB
    
    Args:
        data_dir: Root directory containing class subfolders
        categories: List of class names (subdirectory names)
        img_size: Target image size (height, width)
        max_images_per_class: Limit samples per class (None for all)
        show_progress: Show loading progress bar
        
    Returns:
        Tuple of (X, Y) where:
        - X: Image array of shape (N, H, W, C), dtype float32
        - Y: Label array of shape (N,), dtype int64
        
    Raises:
        DataIntegrityError: If no valid images are loaded
    """
    data_dir = Path(data_dir)
    X, Y = [], []
    
    iterator = tqdm(enumerate(categories), total=len(categories), 
                    desc="Loading") if show_progress else enumerate(categories)
    
    corrupted_count = 0
    
    for class_idx, class_name in iterator:
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            raise DataIntegrityError(f"Class directory not found: {class_dir}")
        
        # Get image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])
        
        if max_images_per_class is not None:
            files = files[:max_images_per_class]
        
        for file_path in files:
            img = load_image(file_path, img_size)
            
            if img is not None:
                X.append(img)
                Y.append(class_idx)
            else:
                corrupted_count += 1
    
    if len(X) == 0:
        raise DataIntegrityError(
            "No valid images loaded. Check data directory and file formats."
        )
    
    if corrupted_count > 0:
        logger.warning(f"Skipped {corrupted_count} corrupted/unreadable images")
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int64)
    
    logger.info(f"Loaded {len(X)} images across {len(categories)} classes")
    
    return X, Y


def split_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
    stratify: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into train, validation, and test sets.
    
    Splitting Strategy:
    -------------------
    1. Stratified splitting to maintain class distribution
    2. Fixed random seed for reproducibility
    3. Two-stage split: first separate test, then split remaining into train/val
    
    Default Split:
    - Training: 70%
    - Validation: 15%
    - Test: 15%
    
    Args:
        X: Feature array
        Y: Label array
        test_size: Fraction for test set
        val_size: Fraction for validation set
        seed: Random seed for reproducibility
        stratify: Use stratified splitting
        
    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing (X, Y) tuple
        
    Example:
        >>> splits = split_dataset(X, Y)
        >>> X_train, y_train = splits['train']
        >>> X_val, y_val = splits['val']
        >>> X_test, y_test = splits['test']
    """
    stratify_arr = Y if stratify else None
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_arr
    )
    
    # Second split: separate validation from training
    # Adjust val_size to account for reduced dataset
    adjusted_val_size = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=seed,
        stratify=stratify_temp
    )
    
    logger.info(
        f"Dataset split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }


def create_tf_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    prefetch: bool = True,
    num_parallel_calls: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """
    Create optimized TensorFlow dataset for GPU training.
    
    Optimization Techniques:
    ------------------------
    1. Prefetching: Overlap data preprocessing with model execution
    2. Parallel mapping: Use multiple CPU cores for preprocessing
    3. Caching: Cache dataset in memory for faster subsequent epochs
    4. Shuffle buffer: Randomize sample order for better generalization
    
    GPU-Optimized Pipeline:
    -----------------------
    The pipeline is designed to keep the GPU fully utilized by:
    - Prefetching next batch while GPU processes current batch
    - Using AUTOTUNE to automatically determine optimal parallelism
    - Batching before augmentation for efficient vectorized operations
    
    Args:
        X: Feature array
        Y: Label array
        batch_size: Samples per batch
        shuffle: Enable shuffling
        augment: Apply data augmentation
        prefetch: Enable prefetching for GPU efficiency
        num_parallel_calls: Parallelism for map operations
        
    Returns:
        tf.data.Dataset configured for efficient training
        
    Note:
        For training, use shuffle=True, augment=True.
        For validation/test, use shuffle=False, augment=False.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    
    if shuffle:
        # Shuffle with buffer size for good randomization
        buffer_size = min(len(X), 10000)
        dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)
    
    # Batch before augmentation for efficiency
    dataset = dataset.batch(batch_size)
    
    if augment:
        dataset = dataset.map(
            augment_batch,
            num_parallel_calls=num_parallel_calls
        )
    
    # Cache after augmentation selection but before prefetch
    dataset = dataset.cache()
    
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


@tf.function
def augment_batch(images: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Apply data augmentation to a batch of images.
    
    Augmentation Strategy for RF Spectrograms:
    -------------------------------------------
    Unlike natural images, RF spectrograms have specific characteristics:
    - Time axis (horizontal): Can be slightly shifted
    - Frequency axis (vertical): Should preserve spectral structure
    - Intensity: Can be slightly varied (noise simulation)
    
    Applied Augmentations:
    1. Random horizontal flip (time reversal - valid for RF)
    2. Random brightness (simulates gain variations)
    3. Random contrast (simulates receiver characteristics)
    
    NOT Applied (would destroy spectral integrity):
    - Vertical flip (would invert frequency mapping)
    - Rotation (would mix time and frequency axes)
    - Heavy geometric distortions
    
    Args:
        images: Batch of images [B, H, W, C]
        labels: Batch of labels [B]
        
    Returns:
        Tuple of augmented images and unchanged labels
    """
    # Random horizontal flip (time axis - valid transformation)
    images = tf.image.random_flip_left_right(images)
    
    # Random brightness adjustment (simulates gain variations)
    images = tf.image.random_brightness(images, max_delta=0.1)
    
    # Random contrast adjustment (receiver characteristics)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    
    # Ensure values stay in valid range
    images = tf.clip_by_value(images, 0.0, 1.0)
    
    return images, labels


def get_class_weights(Y: np.ndarray) -> Dict[int, float]:
    """
    Compute class weights for imbalanced datasets.
    
    Uses balanced class weights formula:
        weight_i = n_samples / (n_classes × n_samples_i)
    
    This ensures that minority classes contribute equally to the loss
    as majority classes, preventing the model from ignoring rare classes.
    
    Args:
        Y: Array of class labels
        
    Returns:
        Dictionary mapping class index to weight
        
    Example:
        >>> weights = get_class_weights(y_train)
        >>> model.fit(X, y, class_weight=weights)
    """
    unique_classes, class_counts = np.unique(Y, return_counts=True)
    n_samples = len(Y)
    n_classes = len(unique_classes)
    
    weights = {}
    for cls, count in zip(unique_classes, class_counts):
        weights[int(cls)] = n_samples / (n_classes * count)
    
    return weights


def get_dataset_statistics(
    data_dir: Path,
    categories: List[str]
) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics.
    
    Returns detailed statistics useful for:
    1. Paper methodology section
    2. Data quality verification
    3. Class balance assessment
    
    Args:
        data_dir: Dataset root directory
        categories: List of class names
        
    Returns:
        Dictionary containing:
        - num_classes: Number of classes
        - total_samples: Total sample count
        - samples_per_class: Dict of class counts
        - min_samples: Minimum samples in any class
        - max_samples: Maximum samples in any class
        - mean_samples: Mean samples per class
        - std_samples: Standard deviation of samples per class
        - class_balance_ratio: Ratio of max to min class size
    """
    _, class_counts = validate_dataset_directory(data_dir)
    
    counts = list(class_counts.values())
    
    return {
        'num_classes': len(categories),
        'total_samples': sum(counts),
        'samples_per_class': class_counts,
        'min_samples': min(counts),
        'max_samples': max(counts),
        'mean_samples': np.mean(counts),
        'std_samples': np.std(counts),
        'class_balance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf')
    }
