def test_cpu_vs_gpu():
    import numpy as np
    from training.utils import CPUModel
    import os

    os.environ["KERAS_BACKEND"] = "torch"
    import keras_core as keras

    batch_size = 10

    for model_name in os.listdir("trained_models"):
        path = os.path.join("trained_models", model_name)
        gpu_model: keras.Sequential = keras.models.load_model(path, compile=False)
        # this is needed for some reason, as keras creates a temporary file/folder for model execution, which contains the model name
        # therefore the model can't have some characters in its name which is weird because it works in the notebooks
        gpu_model.name = "test"
        cpu_model = CPUModel(gpu_model)
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
