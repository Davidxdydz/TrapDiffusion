import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
from training.datasets import load_dataset

info, (train_x, train_y), (val_x, val_y) = load_dataset(
    "Multi-Occupation, Multi Isotope, random matrix, normalized"
)

model = keras.models.load_model("final_models/full_model.keras")

prediction = model.predict(val_x[:1])
print(prediction)
