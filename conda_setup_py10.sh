#!/bin/bash

# Fail fast: without this, a failed `conda create`/`activate` silently falls through
# and the rest of the script builds against the *base* conda env.
set -euo pipefail

exec > >(tee -i setup_log_py10.txt) 2>&1

# get working directiry for where crest repo is located (assumes this script is run from the repo root)
CREST_SPARSE_DIR=$(pwd)
echo "Working directory: $CREST_SPARSE_DIR"
# set root dir for git repos to be one level back from the crest repo
GIT_REPOS_DIR="$CREST_SPARSE_DIR/.."
echo "Git repos directory: $GIT_REPOS_DIR"

# clean up old builds
rm -rf "$CREST_SPARSE_DIR/build"
rm -rf "$GIT_REPOS_DIR/gtsam"

# Make `conda activate` available as a shell function before we use it
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate conda environment (used instead of venv for isolation)
echo "Creating and activating conda environment 'crest'..."
# check if the environment already exists
if conda info --envs | grep -q "crest_py10"; then
    echo "Conda environment 'crest_py10' already exists. Removing it..."
    conda env remove -n crest_py10 -y
fi

# --override-channels keeps us off repo.anaconda.com/pkgs/{main,r}, which conda >=25
# refuses to use non-interactively until their Terms of Service are accepted. It also
# keeps the whole env on a single channel, avoiding conda-forge/defaults ABI mismatches.
echo "Conda environment 'crest_py10' does not exist. Creating it..."
conda create -n crest_py10 -c conda-forge --override-channels python=3.10 -y
conda activate crest_py10

# Refuse to continue if we are not actually inside crest_py10 -- otherwise every
# $CONDA_PREFIX below silently points at the base env and pollutes it.
if [ "$(basename "${CONDA_PREFIX:-}")" != "crest_py10" ]; then
    echo "ERROR: expected to be in the 'crest_py10' env, but CONDA_PREFIX='${CONDA_PREFIX:-<unset>}'" >&2
    exit 1
fi
echo "Active conda env: $CONDA_PREFIX"

# Install C++ build dependencies via conda
echo "Installing C++ build dependencies via conda..."
# conda install -c conda-forge cmake eigen pybind11 boost libgomp openvdb -y
# conda install -c conda-forge cmake eigen pybind11 boost libgomp openvdb tbb-devel gcc_linux-64 gxx_linux-64 -y
# gcc pinned to 13: unpinned resolves to GCC 16, whose more aggressive
# -Wmaybe-uninitialized false-positives on Eigen code and trips GTSAM's -Werror.
# 13.x also matches the toolchain conda-forge builds openvdb/boost with.
# `pip` is explicit: conda-forge's python package does not depend on pip, so a
# bare `conda create python=3.10` env has no pip at all (line ~92 needs it).
# `zlib` (not just the libzlib runtime conda pulls in automatically) is required:
# OpenVDB's FindOpenVDB.cmake does find_package(ZLIB), which needs libz.so + zlib.h
# inside the prefix -- the conda toolchain will not link the system /usr/lib copy.
conda install -c conda-forge --override-channels pip cmake eigen pybind11 boost libgomp openvdb tbb-devel suitesparse zlib gcc_linux-64=13 gxx_linux-64=13 openssh git -y

# Build/Install GTSAM (into conda prefix so it stays isolated from system)
echo "Cloning and building GTSAM..."
cd "$GIT_REPOS_DIR"
# Forked GTSAM 4.3a1 carrying our constrained-module heap-overflow fix
# (NonlinearEquality/InequalityConstraint::violationVector: middleCols -> segment).
git clone https://github.com/joeliechty/gtsam.git
# git clone git@github.com:joeliechty/gtsam.git   # SSH alternative
cd gtsam
git checkout release-4.3a1-fixes  # 4.3a1 + constrained-module fixes
mkdir -p build
cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_WITH_TBB=ON
make -j8
make install

# Make GTSAM libraries visible to the dynamic linker within the conda env
echo "Setting LD_LIBRARY_PATH for GTSAM libraries..."
# ${VAR:-} guards against `set -u` aborting when LD_LIBRARY_PATH is unset (the usual case)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Build/Install CREST-sparse
echo "Building and installing CREST-sparse..."
cd "$CREST_SPARSE_DIR"

# CMakeLists.txt resolves GTSAM/OpenVDB relative to the Python interpreter's prefix,
# so a stray system/base python here means find_package() looks in the wrong place.
echo "Using python: $(command -v python)  ($(python --version 2>&1))"
echo "Using cmake:  $(command -v cmake)"
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: python is not the crest_py10 interpreter" >&2
    exit 1
fi

# Runtime dependencies are declared in pyproject.toml; installing the package
# brings them in. [viz,web] adds the PyVista windows the demo scripts open and
# the viser workbench.
pip install ".[viz,web]" -v

