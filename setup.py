from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "pyboostgraph",
        ["pyboostgraph.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-O3", "-fopenmp"],  # Enable OpenMP
        extra_link_args=["-fopenmp"],  # Link OpenMP library
    ),
]

setup(
    name="pyboostgraph",
    version="0.13",
    ext_modules=ext_modules,
)
