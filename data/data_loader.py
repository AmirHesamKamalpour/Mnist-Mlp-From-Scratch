from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np


ArrayTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_dataset(file_path: str | Path, verbose: bool = True) -> ArrayTuple:
    """
    Load the pickled MNIST dataset used in the notebook.

    Expected pickle structure:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")

    train_set, valid_set, test_set = data
    X_train, y_train = train_set
    X_val, y_val = valid_set
    X_test, y_test = test_set

    if verbose:
        print("MNIST dataset loaded successfully:")
        print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
        print(f"X_val:   {X_val.shape} | y_val:   {y_val.shape}")
        print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def check_flattened_shapes(*arrays: np.ndarray) -> int:
    """
    Confirm that all arrays contain the same number of features.
    Returns the shared feature count.
    """
    if not arrays:
        raise ValueError("At least one array is required.")

    for i, X in enumerate(arrays, start=1):
        print(f"Array {i}: {X.shape[0]} samples, {X.shape[1]} features per sample")

    num_features = arrays[0].shape[1]
    if not all(X.shape[1] == num_features for X in arrays):
        raise ValueError("Dimension mismatch detected across arrays.")

    return num_features


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/validation split using NumPy only.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    if random_seed is not None:
        np.random.seed(random_seed)

    classes = np.unique(y)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = int(len(cls_indices) * val_fraction)

        val_indices.extend(cls_indices[:n_val].tolist())
        train_indices.extend(cls_indices[n_val:].tolist())

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    return X_train, X_val, y_train, y_val


def rebuild_notebook_split(
    file_path: str | Path,
    val_fraction: float = 0.1,
    random_seed: int = 42,
    verbose: bool = True,
) -> ArrayTuple:
    """
    Reproduce the notebook's workflow:
    1) load train/val/test from mnist.pkl
    2) concatenate original train + val
    3) create a fresh stratified train/val split
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(file_path, verbose=verbose)

    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    X_train_new, X_val_new, y_train_new, y_val_new = train_val_split(
        X_full,
        y_full,
        val_fraction=val_fraction,
        random_seed=random_seed,
    )

    if verbose:
        print(f"Rebuilt split -> train: {X_train_new.shape}, val: {X_val_new.shape}")

    return X_train_new, y_train_new, X_val_new, y_val_new, X_test, y_test
