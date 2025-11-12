import os

from setuptools import setup, Extension
import pybind11

GTSAM_INCLUDE_DIR = "/usr/local/include/gtsam"
GTSAM_LIBRARY_DIR = "/usr/local/lib"
EIGEN3_INCLUDE_DIR = "/usr/include/eigen3"

ext_modules = [
    Extension(
        "crest_sparse",
        sources=[
            "src/bindings.cpp",
            "src/gtsam_factors.cpp",
            "src/cosserat_rod.cpp",
            "src/cosserat_rod_solver.cpp",
            "src/parallel_robot.cpp"
        ],
        include_dirs=[
            "include",
            pybind11.get_include(),
            GTSAM_INCLUDE_DIR,
            EIGEN3_INCLUDE_DIR
        ],
        library_dirs=[GTSAM_LIBRARY_DIR],
        libraries=["gtsam"],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3", "-fPIC"],
    ),
]

setup(
    name="crest_sparse",
    version="0.1",
    author="James Ferguson",
    author_email="todo@example.com",
    description="Fast continuum robot state estimation using sparse nonlinear optimization.",
    ext_modules=ext_modules,
    zip_safe=False,
)