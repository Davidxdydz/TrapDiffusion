import numpy as np
from models.cpu import CPUSequential
import time
import keras
from keras import ops
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import datetime


def estimate_model_time(model, batch_size=1000, average=10):
    """
    Inference time in seconds per sample
    """
    x = np.random.uniform(size=(batch_size, model.input_shape[1]))
    x = ops.convert_to_tensor(x)
    start = time.perf_counter()
    for _ in range(average):
        model.predict(x, batch_size=batch_size, verbose=0)
    end = time.perf_counter()
    return (end - start) / average / batch_size


def estimate_cpu_time(model, batch_size=1000, average=10):
    """
    Inference time in seconds per sample
    """
    # cpu_model = CPUSequential(model)
    cpu_model = model.cpu()
    return estimate_model_time(cpu_model, batch_size, average)


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


class CustomTensorboard(keras.callbacks.Callback):
    def __init__(self, log_dir="logs", name=None):
        self.log_dir = Path(log_dir)
        if name is None:
            self.name = f"{datetime.datetime.now():%y-%m-%d %H-%M-%S}"
        else:
            self.name = name
        self.sw = SummaryWriter(self.log_dir / self.name)

    def on_epoch_end(self, epoch, logs=None):
        for metric, value in logs.items():
            self.sw.add_scalar(metric, value, epoch)
        self.sw.add_scalar(
            "learning_rate", self.model.optimizer.get_config()["learning_rate"], epoch
        )


def manual_scheduler(epoch, lr):
    if epoch == 0:
        with open("lr.txt", "w") as f:
            f.write(f"{lr:.2e}")
        return lr
    else:
        with open("lr.txt") as f:
            try:
                lr = float(f.read())
            except:
                pass
        return lr
