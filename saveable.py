import functools
from pathlib import Path
import matplotlib.pyplot as plt


def saveable(savename_func=None, default_dir="report/figures"):
    def saveable_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dir = kwargs.pop("dir", None)
            savename = kwargs.pop("savename", None)
            save = kwargs.pop("save", False)
            width_scale = kwargs.pop("width_scale", 1)
            height_scale = kwargs.pop("height_scale", 1)
            (w, h) = plt.rcParams["figure.figsize"]
            if "height" in kwargs:
                h = kwargs.pop("height")
            if "width" in kwargs:
                w = kwargs.pop("width")
            plt.figure(figsize=(width_scale * w, height_scale * h))
            result = func(*args, **kwargs)
            if savename is not None or (dir != default_dir and dir is not None):
                save = True
            if save:
                if dir is None:
                    dir = default_dir
                path = Path(dir)
                path.mkdir(parents=True, exist_ok=True)
                if savename is None:
                    if savename_func is None:
                        savename = func.__name__
                    else:
                        savename = savename_func(**kwargs)
                path /= savename
                path = path.with_suffix(".pdf")
                plt.savefig(path, bbox_inches="tight")
                print(f"Saved figure to {path}")
            return result

        return wrapper

    return saveable_decorator
