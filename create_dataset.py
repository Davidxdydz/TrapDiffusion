from models.analytical.datasets import create_dataset
from models.analytical.trapdiffusion import SingleOccupationSingleIsotope
import argparse

models = {
    "SISO": SingleOccupationSingleIsotope,
}

presets = {
    "SISO_fixed": {
        "model": "SISO",
        "dataset_name": "Single-Occupation, Single Isotope, fixed matrix",
        "configs": 1,
        "initial_per_config": 10000,
        "n_timesteps": 100,
        "include_params": False,
        "seed": 1,
        "dir": "datasets",
    },
    "SISO_random": {
        "model": "SISO",
        "dataset_name": "Single-Occupation, Single Isotope, random matrix",
        "configs": 1000,
        "initial_per_config": 100,
        "n_timesteps": 50,
        "include_params": True,
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
