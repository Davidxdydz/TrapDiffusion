from models.analytical import TrapDiffusion
import time
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from models.cpu import CPUSequential
import os
import pandas as pd
import itertools


def plot_color_legend(color_dict, title=None):
    for k, v in color_dict.items():
        plt.scatter([], [], color=v, label=k)
    plt.legend(title=title)


def plot_df(df: pd.DataFrame, x: str, y: str, c: str = None, **kwargs):
    custom_colors = False
    col = None
    if c is not None:
        if df.dtypes[c].name == "object":
            custom_colors = True
            available_colors = itertools.cycle(
                ["red", "green", "blue", "black", "purple", "orange"]
            )
            colors = {}

            def map_colors(x):
                if x not in colors:
                    colors[x] = next(available_colors)
                return colors[x]

            col = df[c].map(map_colors)
        else:
            col = df[c]
    df.plot(
        x=x,
        y=y,
        c=col,
        kind="scatter",
        grid=True,
        cmap=None if (custom_colors or c is None) else "viridis",
        **kwargs,
    )
    if custom_colors:
        plot_color_legend(colors, c)


def measure_performance(
    analytical_model: type[TrapDiffusion],
    surrogate_model: CPUSequential,
    batch_size=1,
    average=10,
):
    # exclude batch dimension with [1:]
    inputs = np.random.uniform(0, 1, (batch_size, *surrogate_model.input_shape[1:]))
    start = time.perf_counter()
    for _ in range(average):
        predictions = surrogate_model.predict(inputs)
        results = analytical_model.targets_reverse_transform(predictions)
    end = time.perf_counter()
    t = (end - start) / average / batch_size
    return t


def performance_batch_size_plot(analytical_model: type[TrapDiffusion], surrogate_model):
    factors = {"s": 1, "ms": 1e3, "µs": 1e6, "ns": 1e9}
    batch_sizes = np.arange(10, 1000, 10)
    times = []
    for batch_size in tqdm(batch_sizes, leave=False):
        t_s = measure_performance(
            analytical_model, surrogate_model, batch_size=batch_size, average=10
        )
        times.append(t_s)

    times = np.array(times)
    unit = "µs"
    min_pos = np.argmin(times)
    best_time = times[min_pos]
    best_batch_size = batch_sizes[min_pos]
    plt.plot(batch_sizes, times * factors[unit])
    plt.title(f"CPU inference\n{surrogate_model.name}\n")
    plt.annotate(
        f"{best_batch_size} @ {best_time*factors[unit]:.2f}{unit}",
        (best_batch_size, best_time * factors[unit]),
        horizontalalignment="center",
        xytext=(best_batch_size + 20, best_time * factors[unit] * 2),
        arrowprops=dict(
            facecolor="black", width=0.5, shrink=0.05, headwidth=5, headlength=5
        ),
    )
    plt.xlabel("Batch size")
    plt.grid()
    plt.ylabel(f"Time per prediction [{unit}]")


import os

os.environ["KERAS_BACKEND"] = "torch"
import keras_tuner as kt
from training.datasets import load_dataset_info
from models.gpu.pinn import HyperModel
from pathlib import Path
import json
import pandas as pd
from training.utils import estimate_cpu_time
from tqdm.auto import tqdm
import keras


def create_tuner(
    n,
    output_name,
    output_dir,
    clear,
    max_time,
    method,
    dataset_name,
    dataset_dir,
):
    tuner_classes = {
        "random": kt.tuners.RandomSearch,
        "bayesian": kt.tuners.BayesianOptimization,
        "hyperband": kt.tuners.Hyperband,
    }
    specifice_args = {
        "random": {"max_trials": n},
        "bayesian": {"max_trials": n},
        "hyperband": {"max_epochs": 100, "hyperband_iterations": n},
    }
    info = load_dataset_info(dataset_name, dataset_dir)
    tuner = tuner_classes[method](
        hypermodel=HyperModel(
            input_channels=info["input_channels"],
            output_channels=info["output_channels"],
            normalizer=info.get("pre_normalized", False),
            max_cpu_time=max_time,
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
        ),
        max_consecutive_failed_trials=10,  # this is also raised when model is to big
        objective=kt.Objective("val_max_ae", "min"),
        overwrite=clear,
        directory=output_dir,
        project_name=output_name,
        **specifice_args[method],
    )
    return tuner


def load_tuner(path):
    args_path = Path(path) / "tuner_args.json"
    tuner_args = json.loads(args_path.read_text())
    return create_tuner(**tuner_args)


def load_model(path, trial_id, reload=False):
    path = Path(path)
    model_path = path / f"trial_{trial_id}" / "model.keras"
    if model_path.exists() and not reload:
        return keras.models.load_model(model_path)
    tuner = load_tuner(path)
    trial = tuner.oracle.get_trial(trial_id)
    model = tuner.load_model(trial)
    model.save(model_path)


def load_search(path, reload=False):
    metrics_path = Path(path) / "metrics.pickle"
    if metrics_path.exists() and not reload:
        return pd.read_pickle(metrics_path)
    tuner = load_tuner(path)
    trials = tuner.oracle.trials
    data = {}
    for id, trial in tqdm(trials.items()):
        all_values = {}
        try:
            all_values.update(trial.hyperparameters.values)
            for metric, history in trial.metrics.metrics.items():
                all_values[metric] = history.get_last_value()
            modelpath = Path(path) / f"trial_{id}" / "model.keras"
            if modelpath.exists() and not reload:
                model = keras.models.load_model(modelpath)
            else:
                model = tuner.load_model(trial)
                model.save(modelpath)
            all_values["cpu_time"] = estimate_cpu_time(model)
            all_values["param_count"] = model.count_params()
            data[id] = all_values
        except Exception as e:
            print(f"Error loading trial {id}: {e}")

    df = pd.DataFrame(data).T.convert_dtypes()
    df.to_pickle(metrics_path)
    return df
