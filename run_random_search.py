import pathlib
import shutil
from training.utils import (
    SearchModelGenerator,
    random_search,
    ParameterRange,
    SearchModel,
)
from evaluation import compress_random_search
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--clear", "-c", action="store_true", default=False)
    parser.add_argument("--max_time", default=50e-6)
    parser.add_argument(
        "--dataset_name",
        default="Single-Occupation, Single Isotope, fixed matrix",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = f"random_search/{args.max_time*1e6:.0f}µs {args.dataset_name}"

    output_dir = pathlib.Path(args.output)
    compressed_file = output_dir.with_suffix(".pkl.gz")

    if args.clear:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        compressed_file.unlink(missing_ok=True)

    def reject(search_model: SearchModel):
        return search_model.cpu_inference > args.max_time  # 50 microseconds

    sosi_fixed_generator = SearchModelGenerator(
        args.dataset_name,
        layer_count=ParameterRange([1, 4]),
        layer_sizes=ParameterRange([16, 32, 64, 128, 256]),
        activations=ParameterRange(["relu", "tanh"]),
        output_activation=ParameterRange(["leaky_relu"]),
        physics_weight=ParameterRange([0.0, 1.0]),
        epochs=ParameterRange([20, 40]),
        reject=reject,
    )

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
