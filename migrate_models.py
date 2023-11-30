import os

os.environ["KERAS_BACKEND"] = "torch"
import keras_core
import keras
import pathlib


def get_keras_files(directory):
    files = []
    for dir, _, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            if file.endswith(".keras"):
                yield os.path.join(dir, file)
    return files


def update_model(path):
    path = pathlib.Path(path)
    old_path = path.with_suffix(".keras.old")
    if old_path.exists():
        print(f".", end="", flush=True)
        return
    try:
        keras_core_model = keras_core.models.load_model(path, compile=False)
        keras_core_config_json = keras_core_model.to_json()
        keras_config_json = keras_core_config_json.replace("keras_core", "keras")
        keras_model = keras.models.model_from_json(keras_config_json)
        keras_model.load_state_dict(keras_core_model.state_dict())
        path.rename(old_path)
        keras_model.save(path)
        print(f"Updated {path}")
    except Exception as e:
        print(f"Failed to update {path}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", default=".", nargs="?")
    args = parser.parse_args()
    for file in get_keras_files(args.directory):
        update_model(file)
