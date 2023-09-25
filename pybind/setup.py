# python setup.py build_ext --inplace
from setuptools import setup, Extension  
import pybind11

functions_module = Extension(  
    name ='inference',  
    sources = ['inference_wrapper.cpp', 'inference.cpp'],
    extra_compile_args = ['/O2'],
    include_dirs = [pybind11.get_include()] 
)
  
setup(ext_modules = [functions_module])
