from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import rebuild_notebook_split
from models.model import NeuralNetwork
from utils.metrics import (
    build_confusion_matrix,
    per_class_accuracy,
    to_labels,
    top_confusion_pairs,
)
from utils.visualization import plot_confusion_matrix, plot_misclassified_examples


def analyze_model_predictions(
    model: NeuralNetwork,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int = 10,
    class_names: list[int] | list[str] | None = None,
    img_shape: tuple[int, int] = (28, 28),
    n_wrong_to_plot: int = 12,
    confusion_top_k: int = 5,
    split_name: str = "Dataset",
    show_plots: bool = True,
) -> dict:
    y_true = to_labels(y)
    y_pred = model.predict(X)

    cm = build_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    class_acc = per_class_accuracy(cm)
    overall_acc = float(np.mean(y_true == y_pred))
    worst_class = int(np.argmin(class_acc))

    worst_row = cm[worst_class].copy()
    worst_row[worst_class] = 0
    most_confused_with = np.argsort(worst_row)[::-1]

    print(f"\n=== {split_name} analysis ===")
    print(f"Overall accuracy: {overall_acc:.4f}\n")
    print("1) Accuracy for each class:")
    for i, acc in enumerate(class_acc):
        label = class_names[i] if class_names is not None else i
        print(f"Class {label}: {acc:.4f} ({acc * 100:.2f}%)")

    worst_label = class_names[worst_class] if class_names is not None else worst_class
    print("\n2) Worst classified class:")
    print(
        f"Class {worst_label} with accuracy "
        f"{class_acc[worst_class]:.4f} ({class_acc[worst_class] * 100:.2f}%)"
    )

    top_wrong = [(j, int(worst_row[j])) for j in most_confused_with[:3] if worst_row[j] > 0]
    if top_wrong:
        print("Most often mistaken as:")
        for j, count in top_wrong:
            pred_label = class_names[j] if class_names is not None else j
            print(f"  predicted as {pred_label}: {count} times")

    print("\n3) Classes mistaken with each other most:")
    top_pairs = top_confusion_pairs(cm, top_k=confusion_top_k, symmetric=True)
    for i, j, count in top_pairs:
        label_i = class_names[i] if class_names is not None else i
        label_j = class_names[j] if class_names is not None else j
        print(f"Classes {label_i} and {label_j}: {count} mutual confusions")

    if show_plots:
        print("\n4) Plotting misclassified examples...")
        plot_misclassified_examples(X, y_true, y_pred, n=n_wrong_to_plot, img_shape=img_shape)
        print("\nConfusion matrix:")
        plot_confusion_matrix(cm, class_names=class_names, title=f"{split_name} Confusion Matrix")

    return {
        "overall_accuracy": overall_acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "per_class_accuracy": class_acc,
        "worst_class": worst_class,
        "top_confusions": top_pairs,
    }


def evaluate_model(
    model_path: str | Path,
    dataset_path: str | Path,
    split: str = "test",
    show_plots: bool = True,
) -> dict:
    model, _ = NeuralNetwork.load(model_path)
    X_train, y_train, X_val, y_val, X_test, y_test = rebuild_notebook_split(dataset_path, verbose=False)

    split = split.lower()
    if split == "train":
        X, y, split_name = X_train, y_train, "Train Set"
    elif split == "val":
        X, y, split_name = X_val, y_val, "Validation Set"
    elif split == "test":
        X, y, split_name = X_test, y_test, "Test Set"
    else:
        raise ValueError("split must be one of: train, val, test")

    results = analyze_model_predictions(
        model,
        X,
        y,
        num_classes=10,
        class_names=list(range(10)),
        img_shape=(28, 28),
        n_wrong_to_plot=12,
        confusion_top_k=5,
        split_name=split_name,
        show_plots=show_plots,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved NumPy MNIST model.")
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data" / "dataset" / "mnist.pkl"),
        help="Path to mnist.pkl",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "saved_models" / "mnist_model.pkl"),
        help="Path to saved model pickle",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        split=args.split,
        show_plots=not args.no_plots,
    )
