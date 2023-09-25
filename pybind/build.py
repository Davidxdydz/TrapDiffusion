# located in /pybind/

import subprocess

subprocess.run(["python", "setup.py","clean", "--all", "build_ext", "--build-lib", "../"])