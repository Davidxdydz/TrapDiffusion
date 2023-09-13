# located in /pybind/

import subprocess
import shutil

subprocess.run(["python", "setup.py","clean", "--all", "build_ext", "--inplace"])
shutil.copy2("inference.cp39-win_amd64.pyd","../")