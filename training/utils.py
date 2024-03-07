import numpy as np
from models.cpu import CPUSequential
import time
import keras
from training.datasets import load_dataset


def estimate_model_time(model, batch=1000, average=10):
    """
    Inference time in seconds per sample
    """
    x = np.random.uniform(size=(batch, model.input_shape[1]))
    start = time.perf_counter()
    for _ in range(average):
        model.predict(x, batch_size=batch, verbose=0)
    end = time.perf_counter()
    return (end - start) / average / batch


def estimate_cpu_time(model, batch=1000, average=10):
    """
    Inference time in seconds per sample
    """
    cpu_model = CPUSequential(model)
    return estimate_model_time(cpu_model, batch, average)


def calculate_metrics(
    model: keras.Model,
    hp,
    dataset_name,
    dataset_dir="datasets",
):
    metrics = dict(
        cpu_time=estimate_cpu_time(model),
        gpu_time=estimate_model_time(model),
        param_count=model.count_params(),
    )
    metrics.update(hp.values)
    # TODO maybe output path
    return metrics
