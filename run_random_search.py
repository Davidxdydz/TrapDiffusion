from models.pinn import SOSIFixed
from training.utils import (
    SearchModelGenerator,
    random_search,
    ParameterRange,
    SearchModel,
)
import matplotlib.pyplot as plt
from evaluation import compress_random_search


def reject(search_model: SearchModel):
    return search_model.cpu_inference > 15e-6  # 15 microseconds


if __name__ == "__main__":
    sosi_fixed_generator = SearchModelGenerator(
        SOSIFixed(),
        layer_count=ParameterRange([1, 4]),
        layer_sizes=ParameterRange([16, 32, 64, 128, 256]),
        activations=ParameterRange(["relu", "tanh"]),
        output_activation=ParameterRange(["leaky_relu"]),
        physics_weight=ParameterRange([0, 0.05]),
        epochs=ParameterRange([20, 40]),
        reject=reject,
    )
    random_search(
        sosi_fixed_generator, 100, "random_search/15µs_leaky_relu", quiet=True
    )
    compress_random_search(
        "random_search/15µs_leaky_relu",
        "random_search/15µs_leaky_relu.pkl.gz",
        quiet=True,
    )
