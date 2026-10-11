
# CLEAN CODED this. it would an insult to write comments for them.
# Namings are self-explanatory. (check README2)

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class RegularizationConfig:
    l2_lambda: float = 0.0
    early_stopping: bool = False
    patience: int = 5
    min_delta: float = 0.0
    restore_best_weights: bool = True


@dataclass
class ArchitectureConfig:
    layer_sizes: list[int]
    activations: list[str]
    weight_inits: Any = "xavier"


@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    momentum: float = 0.9
    shuffle: bool = True


@dataclass
class PreprocessingConfig:
    standardize: bool = False


@dataclass
class ModelConfig:
    architecture: ArchitectureConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    loss: str = "categorical_crossentropy"
    seed: int | None = None


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        architecture=ArchitectureConfig(**data["architecture"]),
        training=TrainingConfig(**data.get("training", {})),
        preprocessing=PreprocessingConfig(**data.get("preprocessing", {})),
        regularization=RegularizationConfig(**data.get("regularization", {})),
        loss=data.get("loss", "categorical_crossentropy"),
        seed=data.get("seed"),
    )


def load_config_json(file_path: str | Path) -> ModelConfig:
    file_path = Path(file_path)
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return model_config_from_dict(data)


def save_config_json(config: ModelConfig, file_path: str | Path) -> None:
    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)


class Initializer:
    @staticmethod
    def create(
        shape: tuple[int, int],
        init_cfg: str | dict[str, Any] | Callable[..., np.ndarray] = "xavier",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        fan_in, fan_out = shape

        if callable(init_cfg):
            W = init_cfg(shape, rng)
            return np.asarray(W, dtype=float)

        if isinstance(init_cfg, str):
            init_cfg = {"type": init_cfg}

        init_type = init_cfg.get("type", "xavier").lower()

        if init_type == "xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=shape)
        if init_type == "he":
            std = np.sqrt(2.0 / fan_in)
            return rng.normal(0.0, std, size=shape)
        if init_type == "normal":
            mean = init_cfg.get("mean", 0.0)
            std = init_cfg.get("std", 0.01)
            return rng.normal(mean, std, size=shape)
        if init_type == "uniform":
            low = init_cfg.get("low", -0.1)
            high = init_cfg.get("high", 0.1)
            return rng.uniform(low, high, size=shape)

        raise ValueError(f"Unknown weight initializer: {init_type}")


class StandardScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class Activation:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z = z
        return np.maximum(0, z)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self.z > 0)


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.a = 1.0 / (1.0 + np.exp(-z))
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.a * (1 - self.a)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        self.a = np.tanh(z)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - self.a**2)


class Linear(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        self.a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return self.a

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output


def get_activation(name: str) -> Activation:
    name = name.lower()
    if name == "relu":
        return ReLU()
    if name == "sigmoid":
        return Sigmoid()
    if name == "tanh":
        return Tanh()
    if name == "linear":
        return Linear()
    if name == "softmax":
        return Softmax()
    raise ValueError(f"Unknown activation: {name}")


class Layer:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[dict[str, np.ndarray]]:
        return []


class Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: Any = "xavier",
        rng: np.random.Generator | None = None,
    ) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.W = Initializer.create((in_features, out_features), weight_init, self.rng)
        self.b = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output: np.ndarray, l2_lambda: float = 0.0) -> np.ndarray:
        batch_size = self.input.shape[0]
        self.dW = (self.input.T @ grad_output) / batch_size

        if l2_lambda > 0:
            self.dW += l2_lambda * self.W

        self.db = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        grad_input = grad_output @ self.W.T
        return grad_input

    def parameters(self) -> list[dict[str, np.ndarray]]:
        return [
            {"param": self.W, "grad": self.dW},
            {"param": self.b, "grad": self.db},
        ]


class ActivationLayer(Layer):
    def __init__(self, activation: str) -> None:
        self.activation_name = activation.lower()
        self.activation = get_activation(activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation.forward(x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return self.activation.backward(grad_output)


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)

        if y_true.ndim == 1:
            self.y_true = np.zeros_like(self.y_pred)
            self.y_true[np.arange(len(y_true)), y_true] = 1.0
        else:
            self.y_true = y_true

        return float(-np.mean(np.sum(self.y_true * np.log(self.y_pred), axis=1)))

    def backward(self) -> np.ndarray:
        return self.y_pred - self.y_true


class MSE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]


def get_loss(name: str) -> Loss:
    name = name.lower()
    if name in ("categorical_crossentropy", "crossentropy", "ce"):
        return CategoricalCrossEntropy()
    if name == "mse":
        return MSE()
    raise ValueError(f"Unknown loss: {name}")


class Optimizer:
    def step(self, layers: list[Layer]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            for p in layer.parameters():
                p["param"] -= self.learning_rate * p["grad"]


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: dict[int, np.ndarray] = {}

    def step(self, layers: list[Layer]) -> None:
        for layer in layers:
            for p in layer.parameters():
                key = id(p["param"])
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(p["param"])

                self.velocity[key] = (
                    self.momentum * self.velocity[key]
                    - self.learning_rate * p["grad"]
                )
                p["param"] += self.velocity[key]


def get_optimizer(name: str, learning_rate: float = 0.01, momentum: float = 0.9) -> Optimizer:
    name = name.lower()
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name in ("sgd_momentum", "momentum"):
        return SGDMomentum(learning_rate=learning_rate, momentum=momentum)
    raise ValueError(f"Unknown optimizer: {name}")


class NeuralNetwork:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.loss_fn = get_loss(config.loss)
        self.optimizer = get_optimizer(
            config.training.optimizer,
            learning_rate=config.training.learning_rate,
            momentum=config.training.momentum,
        )
        self.scaler = StandardScaler() if config.preprocessing.standardize else None
        self.layers = self._build_from_config()
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        arch = self.config.architecture

        if len(arch.layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output layer.")
        if len(arch.activations) != len(arch.layer_sizes) - 1:
            raise ValueError("activations length must equal len(layer_sizes) - 1.")
        if "softmax" in [a.lower() for a in arch.activations[:-1]]:
            raise ValueError("softmax should only appear on the output layer.")
        if arch.activations[-1].lower() == "softmax" and self.config.loss.lower() == "mse":
            raise ValueError(
                "Softmax output is intended to be used with categorical cross-entropy."
            )

    def _get_weight_init_for_layer(self, layer_idx: int) -> Any:
        weight_inits = self.config.architecture.weight_inits

        if isinstance(weight_inits, list):
            expected = len(self.config.architecture.layer_sizes) - 1
            if len(weight_inits) != expected:
                raise ValueError("weight_inits list must have one entry per Dense layer.")
            return weight_inits[layer_idx]

        return weight_inits

    def _build_from_config(self) -> list[Layer]:
        arch = self.config.architecture
        layers: list[Layer] = []

        for i in range(len(arch.layer_sizes) - 1):
            in_features = arch.layer_sizes[i]
            out_features = arch.layer_sizes[i + 1]
            activation = arch.activations[i]
            weight_init = self._get_weight_init_for_layer(i)

            layers.append(
                Dense(
                    in_features=in_features,
                    out_features=out_features,
                    weight_init=weight_init,
                    rng=self.rng,
                )
            )
            layers.append(ActivationLayer(activation))

        return layers

    def _transform_input(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if self.scaler is None:
            return X
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _get_model_state(self) -> list[dict[str, np.ndarray]]:
        state: list[dict[str, np.ndarray]] = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                state.append({"W": layer.W.copy(), "b": layer.b.copy()})
        return state

    def _set_model_state(self, state: list[dict[str, np.ndarray]]) -> None:
        idx = 0
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.W[...] = state[idx]["W"]
                layer.b[...] = state[idx]["b"]
                idx += 1

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_loss: np.ndarray) -> None:
        grad = grad_loss
        l2_lambda = self.config.regularization.l2_lambda

        for layer in reversed(self.layers):
            if isinstance(layer, Dense):
                grad = layer.backward(grad, l2_lambda=l2_lambda)
            else:
                grad = layer.backward(grad)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        X = self._transform_input(X, fit=False)
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        if y.ndim == 2:
            y = np.argmax(y, axis=1)
        return float(np.mean(preds == y))

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = True,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        X_train = self._transform_input(X, fit=True)

        X_val_proc = None
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            X_val_proc = self._transform_input(X_val, fit=False)

        n_samples = X_train.shape[0]
        epochs = self.config.training.epochs
        batch_size = self.config.training.batch_size
        shuffle = self.config.training.shuffle
        reg_cfg = self.config.regularization

        if batch_size <= 0 or batch_size > n_samples:
            batch_size = n_samples

        history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = np.inf
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            if shuffle:
                indices = self.rng.permutation(n_samples)
                X_epoch = X_train[indices]
                y_epoch = y[indices]
            else:
                X_epoch = X_train
                y_epoch = y

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                xb = X_epoch[start:end]
                yb = y_epoch[start:end]

                y_pred = self.forward(xb)
                _ = self.loss_fn.forward(y_pred, yb)
                grad_loss = self.loss_fn.backward()

                self.backward(grad_loss)
                self.optimizer.step(self.layers)

            train_pred = self.forward(X_train)
            train_loss = self.loss_fn.forward(train_pred, y)
            y_train_labels = np.argmax(y, axis=1) if y.ndim == 2 else y
            train_pred_labels = np.argmax(train_pred, axis=1)
            train_acc = float(np.mean(train_pred_labels == y_train_labels))

            history["loss"].append(float(train_loss))
            history["acc"].append(train_acc)

            msg = f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}"

            if X_val_proc is not None and y_val is not None:
                val_pred = self.forward(X_val_proc)
                val_loss = self.loss_fn.forward(val_pred, y_val)
                y_val_labels = np.argmax(y_val, axis=1) if y_val.ndim == 2 else y_val
                val_pred_labels = np.argmax(val_pred, axis=1)
                val_acc = float(np.mean(val_pred_labels == y_val_labels))

                history["val_loss"].append(float(val_loss))
                history["val_acc"].append(val_acc)

                msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"

                if reg_cfg.early_stopping:
                    if val_loss < best_val_loss - reg_cfg.min_delta:
                        best_val_loss = val_loss
                        best_epoch = epoch + 1
                        best_state = self._get_model_state()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

            if verbose:
                print(msg)

            if X_val_proc is not None and y_val is not None and reg_cfg.early_stopping:
                if epochs_without_improvement >= reg_cfg.patience:
                    if reg_cfg.restore_best_weights and best_state is not None:
                        self._set_model_state(best_state)
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1}. "
                            f"Restored best weights from epoch {best_epoch}."
                        )
                    break

        return history

    def save(self, file_path: str | Path, history: dict[str, list[float]] | None = None) -> None:
        model_state = []
        for layer in self.layers:
            if isinstance(layer, Dense):
                model_state.append({"W": layer.W.copy(), "b": layer.b.copy()})

        scaler_state = None
        if self.scaler is not None:
            scaler_state = {
                "mean": None if self.scaler.mean is None else self.scaler.mean.copy(),
                "std": None if self.scaler.std is None else self.scaler.std.copy(),
            }

        payload = {
            "config": asdict(self.config),
            "model_state": model_state,
            "scaler_state": scaler_state,
            "history": history,
        }

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            pickle.dump(payload, f)

        print(f"Model saved to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> tuple["NeuralNetwork", dict[str, list[float]] | None]:
        file_path = Path(file_path)
        with file_path.open("rb") as f:
            payload = pickle.load(f)

        config = model_config_from_dict(payload["config"])
        model = cls(config)
        model._set_model_state(payload["model_state"])

        scaler_state = payload.get("scaler_state")
        if model.scaler is not None and scaler_state is not None:
            model.scaler.mean = scaler_state["mean"]
            model.scaler.std = scaler_state["std"]

        history = payload.get("history")
        print(f"Model loaded from {file_path}")
        return model, history
