from models.pinn import SOSIFixed
import pathlib
import shutil
from training.utils import (
    SearchModelGenerator,
    random_search,
    ParameterRange,
    SearchModel,
)
import matplotlib.pyplot as plt
from evaluation import compress_random_search
import argparse


def reject(search_model: SearchModel):
    return search_model.cpu_inference > 15e-6  # 15 microseconds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    parser.add_argument("--output", "-o", default="random_search/15µs_leaky_relu")
    parser.add_argument("--clear", "-c", action="store_true", default=False)
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output)
    compressed_file = output_dir.with_suffix(".pkl.gz")
    sosi_fixed_generator = SearchModelGenerator(
        SOSIFixed(),
        layer_count=ParameterRange([1, 4]),
        layer_sizes=ParameterRange([16, 32, 64, 128, 256]),
        activations=ParameterRange(["relu", "tanh"]),
        output_activation=ParameterRange(["leaky_relu"]),
        physics_weight=ParameterRange([0, 1], dtype=float),
        epochs=ParameterRange([20, 40]),
        reject=reject,
    )
    if args.clear:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        compressed_file.unlink(missing_ok=True)

    random_search(
        sosi_fixed_generator,
        args.n,
        output_dir,
        quiet=args.quiet,
    )
    compress_random_search(
        output_dir,
        compressed_file,
        quiet=True,
    )
