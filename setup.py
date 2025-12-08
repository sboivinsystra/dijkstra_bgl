from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "pyboostgraph",
        ["pyboostgraph.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="pyboostgraph",
    version="0.9",
    ext_modules=ext_modules,
)
