import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import keras
from keras.layers import Input, Dense
from training.datasets import load_dataset
from models.gpu.layers import Normalizer
from models.gpu.metrics import MaxAbsoluteError
from training.utils import CustomTensorboard
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

info, (train_x, train_y), (val_x, val_y) = load_dataset(
    "Multi-Occupation, Multi Isotope, random matrix, normalized large"
)

model_random = keras.Sequential(
    [
        Input(shape=(info["input_channels"],)),
        Dense(512, activation="sigmoid"),
        Dense(512, activation="sigmoid"),
        Dense(256, activation="sigmoid"),
        Dense(256, activation="sigmoid"),
        Dense(info["output_channels"], activation="leaky_relu"),
        Normalizer(norm_loss_weight=0.1),
    ]
)
model_random.summary()
model_random.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-3),
    loss="mae",
    metrics=[
        MaxAbsoluteError(),
        "mae",
    ],
)

factor = 1
n = len(train_x) // factor
model_random.fit(
    train_x[:n],
    train_y[:n],
    validation_data=(val_x, val_y),
    batch_size=2**13,
    epochs=300,  # int(factor*100),
    shuffle=True,
    callbacks=[
        CustomTensorboard("logs/momi_random_normalized_new", name="large data"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, min_lr=1e-4, patience=10),
        keras.callbacks.ModelCheckpoint(
            "trained_models/MOMI_random_normalized_large_new.keras",
            save_best_only=True,
            monitor="val_max_ae",
        ),
        # keras.callbacks.LearningRateScheduler(manual_scheduler),
    ],
)

model_random.save("trained_models/MOMI_random_large_end_result_new.keras")
