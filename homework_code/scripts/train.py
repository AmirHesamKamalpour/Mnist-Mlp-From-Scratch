from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import rebuild_notebook_split
from models.model import NeuralNetwork, load_config_json
from utils.metrics import relative_imbalance
from utils.visualization import plot_class_distributions, plot_training_history


def train_model(
    dataset_path: str | Path,
    config_path: str | Path,
    save_path: str | Path,
    history_path: str | Path | None = None,
    show_plots: bool = False,
) -> tuple[NeuralNetwork, dict]:
    config = load_config_json(config_path)
    X_train, y_train, X_val, y_val, X_test, y_test = rebuild_notebook_split(dataset_path, verbose=True)

    print(f"Relative train imbalance: {relative_imbalance(y_train):.4f}%")
    model = NeuralNetwork(config)
    history = model.fit(X_train, y_train, X_val, y_val, verbose=True)
    model.save(save_path, history=history)

    if history_path is not None:
        history_path = Path(history_path)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Final validation accuracy: {model.accuracy(X_val, y_val):.4f}")
    print(f"Test accuracy snapshot: {model.accuracy(X_test, y_test):.4f}")

    if show_plots:
        plot_class_distributions(y_train, y_val, y_test)
        plot_training_history(history)

    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NumPy MNIST network.")
    parser.add_argument(
        "--dataset-path",
        default=str(PROJECT_ROOT / "data" / "dataset" / "mnist.pkl"),
        help="Path to mnist.pkl",
    )
    parser.add_argument(
        "--config-path",
        default=str(PROJECT_ROOT / "config" / "config.json"),
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--save-path",
        default=str(PROJECT_ROOT / "models" / "saved_models" / "mnist_model.pkl"),
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--history-path",
        default=str(PROJECT_ROOT / "models" / "saved_models" / "training_history.json"),
        help="Where to save the training history JSON",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display training plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        dataset_path=args.dataset_path,
        config_path=args.config_path,
        save_path=args.save_path,
        history_path=args.history_path,
        show_plots=args.show_plots,
    )
