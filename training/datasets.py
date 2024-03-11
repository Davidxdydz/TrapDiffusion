import numpy as np
from tqdm.auto import tqdm
import yaml
from pathlib import Path
from typing import Union, Dict
from models.analytical import TrapDiffusion
import time
from datetime import timedelta
import random


def load_dataset_info(dataset_name, dataset_dir):
    """
    Load the dataset info from the dataset directory.
    """
    dataset_dir = Path(dataset_dir)
    dataset_info_file = dataset_dir / dataset_name / "info.yaml"
    with open(dataset_info_file, "r") as f:
        info = yaml.unsafe_load(f)
    return info


def dataset_channels(
    model: type[TrapDiffusion],
    include_params: bool,
    configs: int,
    initial_per_config: int,
    n_timesteps: int,
):
    samples = configs * initial_per_config * n_timesteps
    n_params = 0
    analytical_model = model()
    n_init = len(analytical_model.initial_values())
    if include_params:
        n_params = len(analytical_model.get_relevant_params())
    input_channels = 1 + n_init + n_params
    output_channels = n_init
    correction_channels = n_init
    return samples, input_channels, output_channels, correction_channels


def estimate_dataset_size(
    model: type[TrapDiffusion],
    include_params: bool,
    configs: int,
    initial_per_config: int,
    n_timesteps: int,
):
    """
    Estimate the size of the datase: number of samples, size in MB.
    """
    samples, input_channels, output_channels, correction_channels = dataset_channels(
        model, include_params, configs, initial_per_config, n_timesteps
    )

    # float32 precision is 4
    bytes_per_sample = (input_channels + output_channels + correction_channels) * 4
    return samples, samples * bytes_per_sample


def create_dataset(
    model: type[TrapDiffusion],
    dataset_name,
    dir="datasets",
    configs=1000,
    initial_per_config=100,
    n_timesteps=50,
    include_params=False,
    info: Union[Dict, None] = None,
    seed=None,
    verbose=True,
    pre_normalized=False,
    log_t_eval=False,
):
    total_samples, total_size = estimate_dataset_size(
        model,
        include_params,
        configs,
        initial_per_config,
        n_timesteps,
    )
    start = time.perf_counter()
    print(f"Creating {dataset_name}")
    print(f"Estimated samples: {total_samples}")
    print(f"Estimated size: {total_size/1024**2} MB")

    is_fixed = seed == "fixed"
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    if not is_fixed:
        np.random.seed(seed)
        random.seed(seed)

    samples, input_channels, output_channels, correction_channels = dataset_channels(
        model, include_params, configs, initial_per_config, n_timesteps
    )
    x = np.empty((samples, input_channels), dtype=np.float32)
    y = np.empty((samples, output_channels), dtype=np.float32)
    c = np.empty((samples, correction_channels), dtype=np.float32)
    current_index = 0
    for _ in tqdm(range(configs), desc="configs", disable=configs == 1 or not verbose):
        analytical_model = model(fixed=is_fixed)
        for _ in tqdm(
            range(initial_per_config),
            desc="initial_values",
            disable=configs > 1 or not verbose,
        ):
            x_, y_ = analytical_model.training_data(
                n_eval=n_timesteps, include_params=include_params, log_t_eval=log_t_eval
            )
            n = len(x_)
            c_ = analytical_model.correction_factors()
            x[current_index : current_index + n] = x_
            y[current_index : current_index + n] = y_
            c[current_index : current_index + n] = c_
            current_index += n

    if pre_normalized:
        y *= c
    dir = Path(dir)
    dir.mkdir(exist_ok=True)
    dataset_dir = dir.joinpath(dataset_name)
    dataset_dir.mkdir(exist_ok=True)
    x_path = Path("x.npy")
    y_path = Path("y.npy")
    c_path = Path("c.npy")
    np.save(dataset_dir / x_path, x)
    np.save(dataset_dir / y_path, y)
    np.save(dataset_dir / c_path, c)
    if info is None:
        info = dict()
    info["inputs_path"] = x_path
    info["targets_path"] = y_path
    info["corrections_path"] = c_path
    info["configs"] = configs
    info["initial_per_params"] = initial_per_config
    info["n_timesteps"] = n_timesteps
    info["include_params"] = include_params
    info["input_channels"] = x.shape[-1]
    info["output_channels"] = y.shape[-1]
    info["x"] = "t, c_s, c_t_1, c_t_2" + (", relevant_params" if include_params else "")
    info["y"] = (
        "solute and trap concentration at time t"
        if not pre_normalized
        else "solute and trap concentrations at time t normalized by corrections, sum to 1"
    )
    info["seed"] = seed
    info["pre_normalized"] = pre_normalized
    with open(dataset_dir.joinpath("info.yaml"), "w") as f:
        yaml.dump(info, f)
    end = time.perf_counter()
    print(f"Created {dataset_name} in {timedelta(seconds = end- start)}")


def load_dataset(name, dir="datasets", split=0.95):
    dir = Path(dir)
    dataset_dir = dir.joinpath(name)
    with open(dataset_dir.joinpath("info.yaml"), "r") as f:
        info: dict = yaml.unsafe_load(f)
    x: np.ndarray = np.load(dataset_dir / info["inputs_path"])
    y: np.ndarray = np.load(dataset_dir / info["targets_path"])
    c: np.ndarray = np.load(dataset_dir / info["corrections_path"])
    train_samples = int(len(x) * split)
    pre_normalized = info.get("pre_normalized", False)
    if not pre_normalized:
        # append the correction factors to the labels, to be used in the loss function
        y = np.append(y, c, axis=1)
    x_train = x[:train_samples]
    y_train = y[:train_samples]
    x_val = x[train_samples:]
    y_val = y[train_samples:]
    return (
        info,
        (x_train, y_train),
        (x_val, y_val),
    )
