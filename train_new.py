import os

os.environ["KERAS_BACKEND"] = "torch"
from keras.layers import Input, Dense
import keras
from training.datasets import load_dataset
from models.gpu.layers import IsotopeNormalizer, ConcentrationRatios
from models.gpu.metrics import MaxAbsoluteError
from training.utils import CustomTensorboard
from keras import ops


info, (train_x, train_y), (val_x, val_y) = load_dataset(
    "Multi-Occupation, Multi Isotope, random matrix, normalized, log"
)
train_x = ops.array(train_x)
train_y = ops.array(train_y)
val_x = ops.array(val_x)
val_y = ops.array(val_y)
print(val_y.device)

# model1: 1e-3
# model2: 0.5e-3
# model3: 2e-3
# model4: 1e-3: 512-> 500
# model5: 1e-3: 1024-> 1000

name = "model5"


ratio_normalizer = IsotopeNormalizer(name="normed")
cr = ConcentrationRatios()

inputs = Input(shape=(info["input_channels"],))
inputs2 = Input(shape=(info["input_channels"],))


x = Dense(10200, activation="tanh")(inputs2)
x = Dense(512, activation="relu")(x)
x = Dense(512, activation="sigmoid")(x)
x = Dense(1000, activation="leaky_relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(info["output_channels"], activation="sigmoid")(x)

normed_weight = 1 / 3
unnormed_weight = 1 - normed_weight
mlp = keras.Model(inputs=inputs2, outputs=x, name="unnormed")
ratios = cr(inputs)
unnormed = mlp(inputs)
outputs = ratio_normalizer(unnormed, ratios)
model = keras.Model(
    inputs=inputs, outputs={"unnormed": unnormed, "normed": outputs}, name=name
)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "unnormed": "mse",
        "normed": "mae",
    },
    metrics={
        "unnormed": MaxAbsoluteError(),
        "normed": MaxAbsoluteError(),
    },
    loss_weights={
        "unnormed": unnormed_weight,
        "normed": normed_weight,
    },
)


model.fit(
    train_x,
    {"normed": train_y, "unnormed": train_y},
    validation_data=(val_x, {"normed": val_y, "unnormed": val_y}),
    batch_size=2**14,
    validation_batch_size=2**16,
    epochs=100,  # int(factor*100),
    shuffle=True,
    initial_epoch=0,
    callbacks=[
        CustomTensorboard("logs/final", name=model.name),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=1e-4, patience=10),
        keras.callbacks.ModelCheckpoint(
            f"final_models/{model.name}.keras",
            save_best_only=True,
            monitor="val_normed_max_ae",
        ),
    ],
)
