from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate import evaluate_model
from scripts.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full MNIST pipeline.")
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
        "--model-path",
        default=str(PROJECT_ROOT / "models" / "saved_models" / "mnist_model.pkl"),
        help="Where to save / load the model",
    )
    parser.add_argument(
        "--history-path",
        default=str(PROJECT_ROOT / "models" / "saved_models" / "training_history.json"),
        help="Where to save training history",
    )
    parser.add_argument("--skip-train", action="store_true", help="Use an existing saved model")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib plots")
    parser.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.skip_train:
        train_model(
            dataset_path=args.dataset_path,
            config_path=args.config_path,
            save_path=args.model_path,
            history_path=args.history_path,
            show_plots=args.show_plots,
        )

    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        split=args.eval_split,
        show_plots=args.show_plots,
    )
