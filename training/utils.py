from tqdm.auto import tqdm
import yaml
import pathlib
import keras_core as keras
import random
from models.pinn import ModelBuilder
import numpy as np
import time
from training.datasets import load_dataset_info, load_dataset
import os
import pandas as pd
import itertools

import itertools
import matplotlib.pyplot as plt


class CPUModel:
    @staticmethod
    def extract(pinn: keras.Sequential):
        weights = []
        biases = []
        activations = []
        for layer in pinn.layers:
            weights.append(layer.get_weights()[0])
            biases.append(layer.get_weights()[1])
            activations.append(layer.activation.__name__)
        return weights, biases, activations

    def __init__(self, model: keras.Sequential):
        self.weights, self.biases, self.activations = CPUModel.extract(model)
        self.input_shape = model.input_shape
        self.output_shape = model.output_shape
        self.name = model.name

    def predict(self, x):
        for layer, bias, activation in zip(self.weights, self.biases, self.activations):
            x = x @ layer + bias
            if activation == "relu":
                x[x < 0] = 0
            elif activation == "linear":
                ...
            elif activation == "tanh":
                x = np.tanh(x)
            elif activation == "leaky_relu":
                x[x < 0] = 0.3 * x[x < 0]
            elif activation == "sigmoid":
                x = 1 / (1 + np.exp(-x))
            else:
                raise NotImplementedError(f"Activation {activation} not implemented")
        return x


class ParameterRange:
    def __init__(self, choices, discrete=None, dtype=None, name=None):
        """
        choices: eiter a list of possible values or a tuple of (lower, upper) bounds
        k: number of values to sample
        discrete: choices are choices or bounds
        dtype: int, float or str
        """
        if all(map(lambda x: isinstance(x, int), choices)):
            dtype = int
        elif any(map(lambda x: isinstance(x, float), choices)):
            dtype = float
        elif all(map(lambda x: isinstance(x, str), choices)):
            dtype = str
        if discrete is None:
            if len(choices) != 2 or dtype == str:
                discrete = True
            else:
                discrete = False
        self.name = name
        self.choices = choices
        self.discrete = discrete
        self.dtype = dtype
        if not discrete:
            if len(choices) != 2:
                raise ValueError(
                    "Continuous parameters must have exactly two choices: lower and upper bound"
                )
            if dtype not in [int, float]:
                raise ValueError("Continuous parameters must have dtype int or float")

    def info(self):
        return dict(
            name=self.name,
            choices=self.choices,
            discrete=self.discrete,
            dtype=str(self.dtype),
        )

    def random(self, k=None):
        if self.discrete:
            if k == None:
                return random.choice(self.choices)
            else:
                return random.choices(self.choices, k=k)
        else:
            if k == None:
                if self.dtype == float:
                    return random.uniform(*self.choices)
                elif self.dtype == int:
                    return random.randint(*self.choices)
            else:
                result = []
                for _ in range(k):
                    if self.dtype == float:
                        result.append(random.uniform(*self.choices))
                    elif self.dtype == int:
                        result.append(random.randint(*self.choices))
                return result


class SearchModel:
    model: keras.Sequential

    def __init__(
        self,
        model_builder: ModelBuilder,
        layer_sizes,
        activations,
        output_activation,
        physics_weight,
        epochs,
        learning_rate,
        batch_size,
        ReduceLROnPlateau_patience,
        ReduceLROnPlateau_factor,
        EarlyStopping_patience,
        layer_count=None,
        gpu_inference=None,
        cpu_inference=None,
        max_mae=None,
        mean_mae=None,
        total_mae=None,
        save_name=None,
        output_path=None,
        param_count=None,
        model=None,
        dataset_name=None,
        dataset_dir=None,
    ):
        if model_builder is not None:
            self.dataset_name = model_builder.dataset_name
            self.dataset_dir = model_builder.dataset_dir
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir
        if dataset_name is not None:
            self.dataset_name = dataset_name
        self.model_builder = model_builder
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.output_activation = output_activation
        self.physics_weight = physics_weight
        self.ReduceLROnPlateau_patience = ReduceLROnPlateau_patience
        self.ReduceLROnPlateau_factor = ReduceLROnPlateau_factor
        self.EarlyStopping_patience = EarlyStopping_patience
        self.gpu_inference = gpu_inference
        self.cpu_inference = cpu_inference
        self.max_mae = max_mae
        self.mean_mae = mean_mae
        self.total_mae = total_mae
        self.save_name = save_name
        self.output_path = output_path
        self.param_count = param_count
        self.model = model
        self.data = None
        self.layer_count = layer_count

        if self.model is None:
            # TODO less shitty way to do this, have to delete data after train and eval
            self.model, self.loss_function, self.data = self.model_builder.prepare(
                self.layer_sizes,
                self.activations,
                self.physics_weight,
                self.output_activation,
            )
        if self.cpu_inference is None:
            self.cpu_inference = self.cpu_bench()
        if self.gpu_inference is None:
            self.gpu_inference = self.gpu_bench()
        if self.layer_count is None:
            self.layer_count = len(self.layer_sizes)
        self.param_count = self.model.count_params()

    @staticmethod
    def from_dir(path):
        path = pathlib.Path(path)
        model = keras.models.load_model(path / "model.keras", compile=False)
        with open(path / "info.yaml") as f:
            info = yaml.safe_load(f)
        return SearchModel(
            model_builder=None,
            **info,
            model=model,
        )

    def info(self):
        return dict(
            dataset_name=self.dataset_name,
            dataset_dir=self.dataset_dir,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            layer_sizes=self.layer_sizes,
            layer_count=self.layer_count,
            activations=self.activations,
            output_activation=self.output_activation,
            physics_weight=self.physics_weight,
            ReduceLROnPlateau_patience=self.ReduceLROnPlateau_patience,
            ReduceLROnPlateau_factor=self.ReduceLROnPlateau_factor,
            EarlyStopping_patience=self.EarlyStopping_patience,
            max_mae=self.max_mae,
            mean_mae=self.mean_mae,
            total_mae=self.total_mae,
            save_name=self.save_name,
            output_path=str(self.output_path),
            param_count=self.param_count,
            gpu_inference=self.gpu_inference,
            cpu_inference=self.cpu_inference,
        )

    def gpu_bench(self, batch=1000, average=10):
        """
        Inference time in seconds per sample
        """
        x = np.random.uniform(size=(batch, self.model.input_shape[1]))
        start = time.perf_counter()
        for _ in range(average):
            self.model.predict(x, batch_size=batch, verbose=0)
        end = time.perf_counter()
        return (end - start) / average / batch

    def cpu_bench(self, batch=1000, average=10):
        """
        Inference time in seconds per sample
        """
        x = np.random.uniform(size=(batch, self.model.input_shape[1]))
        cpu_model = CPUModel(self.model)
        start = time.perf_counter()
        for _ in range(average):
            cpu_model.predict(x)
        end = time.perf_counter()
        return (end - start) / average / batch

    def load_dataset(self):
        return load_dataset(self.dataset_name, self.dataset_dir)

    def train_and_evaluate(self, output_dir="", quiet=False):
        x, y = self.data
        scheduler = keras.callbacks.ReduceLROnPlateau(
            patience=self.ReduceLROnPlateau_patience,
            factor=self.ReduceLROnPlateau_factor,
        )
        early_stopping = keras.callbacks.EarlyStopping(
            patience=self.EarlyStopping_patience,
            restore_best_weights=True,
        )
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.loss_function,
        )
        self.model.fit(
            x,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            validation_split=0.15,
            callbacks=[scheduler, early_stopping],
            verbose=0 if quiet else 1,
        )
        predictions = self.model.predict(x, batch_size=2**12, verbose=0)
        maes = np.mean(np.abs(predictions - y), axis=1)
        self.max_mae = float(np.max(maes))
        self.mean_mae = float(np.mean(maes))
        self.total_mae = float(np.sum(maes))
        self.save_name = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.output_path = pathlib.Path(f"{output_dir}/{self.save_name}").resolve()
        self.output_path.mkdir(exist_ok=True, parents=True)
        self.model.save(self.output_path / "model.keras")
        with open(self.output_path / "info.yaml", "w") as f:
            yaml.safe_dump(self.info(), f)

        # manual memory management in python yeay...
        self.data = None

    def __repr__(self) -> str:
        return f"SearchModel({self.info()})"


class SearchModelGenerator:
    def __init__(
        self,
        model_builder: ModelBuilder,
        layer_count: ParameterRange,
        layer_sizes: ParameterRange,
        activations: ParameterRange,
        output_activation: ParameterRange,
        physics_weight: ParameterRange,
        epochs=ParameterRange([20]),
        learning_rate=ParameterRange([3e-4]),
        batch_size=ParameterRange([2**12]),
        ReduceLROnPlateau_patience=ParameterRange([5]),
        ReduceLROnPlateau_factor=ParameterRange([0.5]),
        EarlyStopping_patience=ParameterRange([6]),
        reject=lambda search_model: False,
        patience=100,
    ):
        self.model_builder = model_builder
        self.epochs_range = epochs
        self.batch_size_range = batch_size
        self.learning_rate_range = learning_rate
        self.layer_sizes_range = layer_sizes
        self.layer_count_range = layer_count
        self.activations_range = activations
        self.output_activation_range = output_activation
        self.physics_weight_range = physics_weight
        self.ReduceLROnPlateau_patience_range = ReduceLROnPlateau_patience
        self.ReduceLROnPlateau_factor_range = ReduceLROnPlateau_factor
        self.EarlyStopping_patience_range = EarlyStopping_patience
        self.dataset_info = load_dataset_info(
            model_builder.dataset_name, model_builder.dataset_dir
        )
        self.reject = reject
        self.patience = patience

    def info(self):
        return dict(
            model_builder=self.model_builder.info(),
            epochs_range=self.epochs_range.info(),
            batch_size_range=self.batch_size_range.info(),
            learning_rate_range=self.learning_rate_range.info(),
            layer_sizes_range=self.layer_sizes_range.info(),
            layer_count_range=self.layer_count_range.info(),
            activations_range=self.activations_range.info(),
            output_activation_range=self.output_activation_range.info(),
            physics_weight_range=self.physics_weight_range.info(),
            ReduceLROnPlateau_patience_range=self.ReduceLROnPlateau_patience_range.info(),
            ReduceLROnPlateau_factor_range=self.ReduceLROnPlateau_factor_range.info(),
            EarlyStopping_patience_range=self.EarlyStopping_patience_range.info(),
            dataset_info=self.dataset_info,
        )

    def random_model(self):
        for _ in range(self.patience):
            layer_count = self.layer_count_range.random()
            model = SearchModel(
                model_builder=self.model_builder,
                epochs=self.epochs_range.random(),
                batch_size=self.batch_size_range.random(),
                learning_rate=self.learning_rate_range.random(),
                layer_sizes=self.layer_sizes_range.random(layer_count),
                activations=self.activations_range.random(layer_count),
                output_activation=self.output_activation_range.random(),
                physics_weight=self.physics_weight_range.random(),
                ReduceLROnPlateau_patience=self.ReduceLROnPlateau_patience_range.random(),
                ReduceLROnPlateau_factor=self.ReduceLROnPlateau_factor_range.random(),
                EarlyStopping_patience=self.EarlyStopping_patience_range.random(),
            )

            if not self.reject(model):
                return model
        raise RuntimeError(f"No suitable model config found in {self.patience} tries.")


def random_search(
    generator: SearchModelGenerator, n: int, output_dir=None, quiet=False
):
    models = []
    if output_dir is None:
        output_dir = (
            pathlib.Path("random_search") / generator.model_builder.dataset_name
        )
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    with open(output_dir / "info.yaml", "w") as f:
        yaml.safe_dump(generator.info(), f)
    finished = len(os.listdir(output_dir)) - 1
    n -= finished
    for _ in tqdm(range(n), disable=quiet):
        model = generator.random_model()
        print(f"Training {model.info()}")
        model.train_and_evaluate(output_dir=output_dir, quiet=quiet)
        models.append(model)
    return models
