from training.datasets import create_dataset
from models.analytical import SingleOccupationSingleIsotope, MultiOccupationMultiIsotope
import argparse

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
        "seed": 1,
        "dir": "datasets",
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
    "MOMI_fixed": {
        "model": "MOMI",
        "dataset_name": "Multi-Occupation, Multi Isotope, fixed matrix",
        "configs": 1,
        "initial_per_config": 10000,
        "n_timesteps": None,
        "include_params": False,
        "seed": 1,
        "dir": "datasets",
    },
}

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
parser.add_argument("--verbose", "-v", type=bool, default=True)

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

if "model" in arg_dict:
    arg_dict["model"] = models[arg_dict["model"]]
else:
    raise ValueError("model must be specified")

create_dataset(**arg_dict)
