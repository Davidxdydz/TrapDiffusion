# python setup.py build_ext --inplace
from setuptools import setup, Extension  
  
functions_module = Extension(  
    name ='inference',  
    sources = ['inference_wrapper.cpp', 'inference.cpp'],
    extra_compile_args = ['/O2'],
    include_dirs = [r'C:\Users\david\AppData\Local\Programs\Python\Python39\Lib\site-packages\pybind11\include']  
)
  
setup(ext_modules = [functions_module])
