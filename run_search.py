import os

os.environ["KERAS_BACKEND"] = "torch"
from training.datasets import load_dataset
import argparse
import json
from pathlib import Path
from evaluation import create_tuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    parser.add_argument("--output_name", "-o", default=None)
    parser.add_argument("--output_dir", "-d", default="random_search")
    parser.add_argument("--clear", "-c", action="store_true", default=False)
    parser.add_argument(
        "--method", default="random", choices=["random", "bayesian", "hyperband"]
    )
    parser.add_argument(
        "--dataset_name",
        default="Single-Occupation, Single Isotope, fixed matrix",
    )
    parser.add_argument("--dataset_dir", default="datasets")
    args = parser.parse_args()
    if args.output_name is None:
        args.output_name = f"{args.dataset_name}"

    info, (x_train, y_train), (x_val, y_val) = load_dataset(
        args.dataset_name, args.dataset_dir
    )

    args_dict = args.__dict__
    quiet = args_dict.pop("quiet")
    tuner = create_tuner(**args_dict)

    args_dict["clear"] = False
    args_path = Path(args.output_dir) / args.output_name / "tuner_args.json"
    args_path.parent.mkdir(parents=True, exist_ok=True)
    args_path.write_text(json.dumps(args_dict))

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        verbose=0 if quiet else 1,
    )
