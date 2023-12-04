from models.analytical import TrapDiffusion
import time
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from training.utils import CPUModel
import os
from training.utils import SearchModel
import pandas as pd
import itertools


def load_random_search(directory, quiet=False) -> pd.DataFrame:
    models = []
    for dir in tqdm(os.listdir(directory), disable=quiet):
        path = os.path.join(directory, dir)
        if os.path.isdir(path):
            model = SearchModel.from_dir(path)
            models.append(model)
    df = pd.DataFrame.from_dict(map(lambda x: x.info(), models))
    df["max_layer_size"] = df["layer_sizes"].map(max)
    for i in range(1, 6):
        df[f"layer_{i}_size"] = df["layer_sizes"].map(
            lambda x: x[i - 1] if len(x) >= i else None
        )
        df[f"layer_{i}_activation"] = df["activations"].map(
            lambda x: x[i - 1] if len(x) >= i else f"no layer {i}"
        )
    return df


def compress_random_search(
    directory: os.PathLike, output_file: os.PathLike, quiet=False
):
    df = load_random_search(directory, quiet=quiet)
    df.to_pickle(output_file, compression="gzip")


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
    surrogate_model: CPUModel,
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
