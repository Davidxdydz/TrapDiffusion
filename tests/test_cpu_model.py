def test_cpu_vs_gpu_random():
    import numpy as np
    import numpy as np
    from models.cpu import CPUSequential
    from models.cpu.layers import CPULayer
    import os
    from models.gpu.layers import Normalizer

    os.environ["KERAS_BACKEND"] = "torch"
    import keras

    # now build a model containing all available cpu layers
    gpu_model = keras.Sequential()
    gpu_model.add(keras.layers.Input(shape=(10,)))
    gpu_model.add(keras.layers.Dense(10, activation="relu"))
    gpu_model.add(Normalizer())

    cpu_model = CPUSequential(gpu_model)

    # check all layers are used
    for layer in CPULayer.cpu_translation:
        assert any(
            isinstance(x, layer) for x in gpu_model.layers
        ), f"Layer {layer} not used in test, test is wrong"

    # check the model is the same
    inputs = np.random.uniform(0, 1, (10, 10)).astype(np.float32)
    gpu_predictions = gpu_model.predict(inputs)
    cpu_predictions = cpu_model.predict(inputs)
    diff = np.abs(gpu_predictions - cpu_predictions)
    assert np.allclose(
        diff,
        0,
        atol=1e-6,
    ), f"max error: {diff.max()}, activations: {cpu_model.activations}"


def test_cpu_vs_gpu_trained():
    import numpy as np
    from models.cpu import CPUSequential
    import os

    os.environ["KERAS_BACKEND"] = "torch"
    import keras

    batch_size = 10

    for model_name in os.listdir("trained_models"):
        path = os.path.join("trained_models", model_name)
        gpu_model: keras.Sequential = keras.models.load_model(path, compile=False)
        # this is needed for some reason, as keras creates a temporary file/folder for model execution, which contains the model name
        # therefore the model can't have some characters in its name which is weird because it works in the notebooks
        gpu_model.name = "test"
        cpu_model = CPUSequential(gpu_model)
        inputs = np.random.uniform(
            0, 1, (batch_size, *cpu_model.input_shape[1:])
        ).astype(np.float32)
        gpu_predictions = gpu_model.predict(inputs)
        cpu_predictions = cpu_model.predict(inputs)
        diff = np.abs(gpu_predictions - cpu_predictions)
        assert np.allclose(
            # TODO look at why only 1e-6
            diff,
            0,
            atol=1e-6,
        ), f"{model_name} failed, max error: {diff.max()}, activations: {cpu_model.activations}"
