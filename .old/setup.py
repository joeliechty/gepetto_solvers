from setuptools import setup
import pybind11
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os


gtsam_include = os.environ.get("GTSAM_INCLUDE_DIR", "/usr/local/include")
gtsam_lib = os.environ.get("GTSAM_LIB_DIR", "/usr/local/lib")

ext_modules = [
    Pybind11Extension(
        "tendon_robot",  # This becomes the import name in Python
        sources=[
            "./src/bindings.cpp",
        ],
        include_dirs=[
            gtsam_include,
            '/usr/include/eigen3',
            "./include",
            pybind11.get_include()
        ],
        library_dirs=[
            gtsam_lib,
        ],
        libraries=[
            "gtsam",
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="tendon_robot",
    version="0.1",
    author="Your Name",
    description="pybind11 wrapper for factor graph-based tendon robot solvers",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
