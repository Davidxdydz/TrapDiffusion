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
            result = func(*args, **kwargs)
            if savename is not None or dir != default_dir:
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
