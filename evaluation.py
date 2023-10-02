from models.analytical import TrapDiffusion
import time
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from training.utils import CPUModel


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
