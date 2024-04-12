from training.datasets import create_dataset
from models.analytical import SingleOccupationSingleIsotope, MultiOccupationMultiIsotope
import argparse
import datetime

models = {
    "SOSI": SingleOccupationSingleIsotope,
    "MOMI": MultiOccupationMultiIsotope,
}

presets = {
    "SOSI_fixed": {
        "model": "SOSI",
        "dataset_name": "Single-Occupation, Single Isotope, fixed matrix",
        "configs": 1,
        "initial_per_config": 10000,
        "n_timesteps": 100,
        "include_params": False,
        "seed": "fixed",
        "dir": "datasets",
    },
    "SOSI_fixed_normalized": {
        "model": "SOSI",
        "dataset_name": "Single-Occupation, Single Isotope, fixed matrix, normalized",
        "configs": 1,
        "initial_per_config": 50000,
        "n_timesteps": 100,
        "include_params": False,
        "seed": "fixed",
        "dir": "datasets",
        "pre_normalized": True,
    },
    "SOSI_random": {
        "model": "SOSI",
        "dataset_name": "Single-Occupation, Single Isotope, random matrix",
        "configs": 1000,
        "initial_per_config": 100,
        "n_timesteps": 50,
        "include_params": True,
        "seed": 1,
        "dir": "datasets",
    },
    "SOSI_random_normalized": {
        "model": "SOSI",
        "dataset_name": "Single-Occupation, Single Isotope, random matrix, normalized",
        "configs": 5000,
        "initial_per_config": 100,
        "n_timesteps": 100,
        "include_params": True,
        "seed": 1,
        "dir": "datasets",
        "pre_normalized": True,
    },
    "MOMI_fixed": {
        "model": "MOMI",
        "dataset_name": "Multi-Occupation, Multi Isotope, fixed matrix",
        "configs": 1,
        "initial_per_config": 10000,
        "n_timesteps": 100,
        "log_t_eval": True,
        "include_params": False,
        "seed": "fixed",
        "dir": "datasets",
    },
    "MOMI_fixed_normalized": {
        "model": "MOMI",
        "dataset_name": "Multi-Occupation, Multi Isotope, fixed matrix, normalized",
        "configs": 1,
        "initial_per_config": 100000,
        "n_timesteps": 100,
        "log_t_eval": True,
        "include_params": False,
        "seed": "fixed",
        "dir": "datasets",
        "pre_normalized": True,
    },
    "MOMI_random": {
        "model": "MOMI",
        "dataset_name": "Multi-Occupation, Multi Isotope, random matrix",
        "configs": 1000,
        "initial_per_config": 100,
        "n_timesteps": 100,
        "log_t_eval": True,
        "include_params": True,
        "seed": 1,
        "dir": "datasets",
    },
    "MOMI_random_normalized": {
        "model": "MOMI",
        "dataset_name": "Multi-Occupation, Multi Isotope, random matrix, normalized",
        "configs": 5000,
        "initial_per_config": 100,
        "n_timesteps": 100,
        "log_t_eval": True,
        "include_params": True,
        "seed": 1,
        "dir": "datasets",
        "pre_normalized": True,
    },
    "test": {
        "model": "MOMI",
        "dataset_name": "test",
        "configs": 50,
        "initial_per_config": 10,
        "n_timesteps": 40,
        "log_t_eval": True,
        "include_params": True,
        "seed": 1,
        "dir": "datasets",
        "pre_normalized": True,
    },
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, choices=presets.keys())
    # suppress default
    parser.add_argument("--dataset_name", type=str, default=argparse.SUPPRESS)
    parser.add_argument(
        "--model", type=str, choices=models.keys(), default=argparse.SUPPRESS
    )
    parser.add_argument("--configs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--initial_per_config", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--n_timesteps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--include_params", type=bool, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--dir", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--quiet", "-q", action=argparse.BooleanOptionalAction)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--chunksize", type=int, default=1)

    args = parser.parse_args()

    arg_dict = {}
    if args.preset is not None:
        arg_dict.update(presets[args.preset])
    arg_dict.update(vars(args))

    if "dataset_name" not in arg_dict:
        raise ValueError("dataset_name must be specified if no preset is used")

    if "preset" in arg_dict:
        # don't pass preset to create_dataset
        del arg_dict["preset"]

    arg_dict["verbose"] = not arg_dict["quiet"]
    del arg_dict["quiet"]

    if "model" in arg_dict:
        arg_dict["model"] = models[arg_dict["model"]]
    else:
        raise ValueError("model must be specified")

    arg_dict["info"] = {
        "created": datetime.datetime.now().isoformat(),
    }

    create_dataset(**arg_dict)
