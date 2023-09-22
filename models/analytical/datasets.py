import numpy as np
from tqdm.auto import tqdm
import yaml
import pathlib
from typing import Union, Dict
from models.analytical.trapdiffusion import TrapDiffusion


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
):
    analytical_model = model()
    n_init = len(analytical_model.initial_values())
    n_params = len(analytical_model.get_relevant_params())
    # t, c_s, c_t_1, c_t_2, <optional> relevant_params
    input_dim = 1 + n_init + (n_params if include_params else 0)
    output_dim = n_init

    # double precision is 8 bytes
    bytes_per_sample = (input_dim + output_dim) * 8

    total_samples = configs * initial_per_config * n_timesteps
    print(f"Estimated size: {total_samples*bytes_per_sample/1024**2} MB")

    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    np.random.seed(seed)
    x, y = [], []
    for _ in tqdm(range(configs), desc="configs", disable=configs == 1):
        analytical_model = model()
        for _ in tqdm(
            range(initial_per_config), desc="initial_values", disable=configs > 1
        ):
            x_, y_ = analytical_model.training_data(
                n_eval=n_timesteps, include_params=include_params
            )
            x.extend(x_)
            y.extend(y_)

    x, y = np.array(x), np.array(y)
    dir = pathlib.Path(dir)
    dir.mkdir(exist_ok=True)
    dataset_dir = dir.joinpath(dataset_name)
    dataset_dir.mkdir(exist_ok=True)
    np.save(dataset_dir.joinpath("x.npy"), x)
    np.save(dataset_dir.joinpath("y.npy"), y)
    if info is None:
        info = dict()
    info["configs"] = configs
    info["initial_per_params"] = initial_per_config
    info["n_timesteps"] = n_timesteps
    info["include_params"] = include_params
    info["input_dim"] = input_dim
    info["x"] = "t, c_s, c_t_1, c_t_2" + (", relevant_params" if include_params else "")
    info["y"] = "c_s, c_t_1, c_t_2 at time t"
    info["seed"] = seed
    with open(dataset_dir.joinpath("info.yaml"), "w") as f:
        yaml.safe_dump(info, f)


def load_dataset(name, dir="datasets"):
    dir = pathlib.Path(dir)
    dataset_dir = dir.joinpath(name)
    x = np.load(dataset_dir.joinpath("x.npy"))
    y = np.load(dataset_dir.joinpath("y.npy"))
    with open(dataset_dir.joinpath("info.yaml"), "r") as f:
        info = yaml.safe_load(f)
    return x, y, info
