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
        "hyperband": {"max_epochs": 30, "hyperband_iterations": n},
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

    df = pd.DataFrame(data).T
    df.to_pickle(metrics_path)
    return df
