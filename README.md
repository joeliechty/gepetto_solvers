# CREST-Sparse: Continuum Robot ESTimation with Sparse Nonlinear Optimization
Factor graph–based solvers for continuum robots and other elastic structures.

## Description

Continuum robot state estimation can be formulated similarly to SLAM, where variables are connected through spatial motion priors and measurement factors.
This repository demonstrates how to construct factor graph representations of the conditional distribution over continuum robot configurations.

We leverage the sparse nonlinear optimization capabilities of GTSAM to efficiently estimate continuum robot states.
For each model, we provide Python bindings for the underlying solver along with a real-time PyVista plotter for visualizing simulations.

## Build/Install GTSAM

The Python bindings dynamically load classes from GTSAM, so GTSAM must first be built and installed from source.

First clone GTSAM and configure the build with CMake:

```bash
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1 # Tested GTSAM version
mkdir build 
cd build
cmake ..
```

At this point, verify that CMake found all required dependencies (e.g., Boost, Eigen, TBB).
Ensure that there are no critical warnings during configuration.
For more information on dependencies, see the GTSAM [installation documentation](https://borglab.github.io/gtsam/install/)

If everything looks good, you can now build and install gtsam, which will take several minutes:

```bash
make -j8
sudo make install
```

This installs GTSAM headers (needed to build CREST-sparse) in `/usr/local/include` and library files (needed to run CREST-sparse) in `/usr/local/lib`.

## Build/Install CREST-sparse

First clone this repository:
```bash
git clone https://github.com/fergujm2/crest-sparse.git
cd crest-sparse
```

Next create and activate a Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the python dependencies required for plotting and simulation:
```bash
pip install -r requirements.txt
```

Now build and install the CREST-sparse Python module from the `C++` classes in `src/`:
```bash
pip install . -v
```

The final output should include `Successfully installed crest_sparse`, meaning that you can now `import crest_sparse` in Python when the virtual environment is active. 

## Run Test Scripts

The module for plotting and testing lives in `python/tests/`, where there are several examples you can run to verify the solvers are working.
Here are a few options:

```bash
cd python

python -m tests.cosserat.test_priors_sim
python -m tests.cosserat.spring_sim
python -m tests.cosserat.shell_sim
python -m tests.cosserat.dynamics_sim
python -m tests.tendon_robot.test_simple
python -m tests.multi_robot.test_jacobian_control
```

You may get an initial error that says something like: `ImportError: libgtsam.so.4: cannot open shared object file: No such file or directory`.
This indicates that the GTSAM library installation directory is not visible to the dynamic linker.
In most cases this can be resolved by running `sudo ldconfig` and then rerunning the script.

When the chosen script runs successfully, a PyVista render window will appear showing real-time solution geometries for the selected model.
Solution metadata is displayed in the upper-right corner, including optimization solve times and related diagnostics.