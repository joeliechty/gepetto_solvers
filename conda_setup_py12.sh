#!/bin/bash

# Fail fast: without this, a failed `conda create`/`activate` silently falls through
# and the rest of the script builds against the *base* conda env.
set -euo pipefail

exec > >(tee -i setup_log_py12.txt) 2>&1

# get working directiry for where gepetto repo is located (assumes this script is run from the repo root)
GEPETTO_SPARSE_DIR=$(pwd)
echo "Working directory: $GEPETTO_SPARSE_DIR"
# set root dir for git repos to be one level back from the gepetto repo
GIT_REPOS_DIR="$GEPETTO_SPARSE_DIR/.."
echo "Git repos directory: $GIT_REPOS_DIR"

# clean up old builds
rm -rf "$GEPETTO_SPARSE_DIR/build"
rm -rf "$GIT_REPOS_DIR/gtsam"

# Make `conda activate` available as a shell function before we use it
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create and activate conda environment (used instead of venv for isolation)
echo "Creating and activating conda environment 'gepetto_py12'..."
# check if the environment already exists
if conda info --envs | grep -q "gepetto_py12"; then
    echo "Conda environment 'gepetto_py12' already exists. Removing it..."
    conda env remove -n gepetto_py12 -y
fi

# --override-channels keeps us off repo.anaconda.com/pkgs/{main,r}, which conda >=25
# refuses to use non-interactively until their Terms of Service are accepted. It also
# keeps the whole env on a single channel, avoiding conda-forge/defaults ABI mismatches.
echo "Conda environment 'gepetto_py12' does not exist. Creating it..."
conda create -n gepetto_py12 -c conda-forge --override-channels python=3.12 -y
conda activate gepetto_py12

# Refuse to continue if we are not actually inside gepetto_py12 -- otherwise every
# $CONDA_PREFIX below silently points at the base env and pollutes it.
if [ "$(basename "${CONDA_PREFIX:-}")" != "gepetto_py12" ]; then
    echo "ERROR: expected to be in the 'gepetto_py12' env, but CONDA_PREFIX='${CONDA_PREFIX:-<unset>}'" >&2
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
# bare `conda create python=3.12` env has no pip at all (line ~92 needs it).
# `zlib` (not just the libzlib runtime conda pulls in automatically) is required:
# OpenVDB's FindOpenVDB.cmake does find_package(ZLIB), which needs libz.so + zlib.h
# inside the prefix -- the conda toolchain will not link the system /usr/lib copy.
# Pinocchio: rigid-body kinematics + analytical derivatives, for URDF-described
# hands (src/hand/kinematics/rigid/).
#
# PINNED TO THE EIGEN-3.4 BUILD, and that pin is load-bearing. GTSAM bakes
# GTSAM_EIGEN_VERSION_WORLD/MAJOR into gtsam/config.h and static-asserts them
# against whatever Eigen an including translation unit sees; our GTSAM is built
# against 3.4. conda-forge's newest pinocchio (4.1) is built against Eigen 5, so
# a single .cpp including both fails to compile outright. The 4.0.0 h9a60d09_0
# build is the newest one against Eigen 3.4. It also pins boost <1.89 -- there is
# no Eigen-3.4 pinocchio built against boost 1.90 -- which is why boost is 1.88
# here; GTSAM links boost by unversioned soname and loads fine against it.
#
# If you bump this, check `conda search -c conda-forge libpinocchio --info` for
# the eigen-abi pin and keep it on whatever major.minor GTSAM was built with.
# The build hash is platform-specific but NOT python-specific: libpinocchio is the
# pure C++ library, so h2844b27_0 is the linux-64 eigen-abi-3.4 build for every
# python here (the mac script pins h9a60d09_0 for the same 4.0.0 version).
# Pinning it makes conda pick pinocchio 4.0.0 h7ab193c_0 / pinocchio-python
# py312h3f83079_0 automatically.
#
# `libboost-devel`, NOT `boost`: conda-forge froze the old `boost` metapackage at
# 1.85.0, and it hard-pins libboost-python-devel ==1.85.0. libpinocchio 4.0.0
# needs libboost >=1.88, so asking for `boost` makes the solve unsatisfiable
# ("nothing provides _python_rc ... pin on python=3.12 is not installable").
# libboost-devel is the current split-package name and tracks 1.88+.
#
# `eigen=3.4` is explicit because conda-forge now ships eigen 5.0.x by default;
# an unpinned `eigen` would either pull 5.0 (breaking the GTSAM config.h static
# assert described above) or silently drag in the eigen-5 pinocchio build.
conda install -c conda-forge --override-channels pip cmake "eigen=3.4" pybind11 libboost-devel libgomp openvdb tbb-devel suitesparse zlib "libpinocchio=4.0.0=h2844b27_0" "pinocchio=4.0.0" gcc_linux-64=13 gxx_linux-64=13 openssh git -y

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

# Build/Install gepetto_solvers
echo "Building and installing gepetto_solvers..."
cd "$GEPETTO_SPARSE_DIR"

# CMakeLists.txt resolves GTSAM/OpenVDB relative to the Python interpreter's prefix,
# so a stray system/base python here means find_package() looks in the wrong place.
echo "Using python: $(command -v python)  ($(python --version 2>&1))"
echo "Using cmake:  $(command -v cmake)"
if [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
    echo "ERROR: python is not the gepetto_py12 interpreter" >&2
    exit 1
fi

# Runtime dependencies are declared in pyproject.toml; installing the package
# brings them in. [viz,web] adds the PyVista windows the demo scripts open and
# the viser workbench.
pip install ".[viz,web]" -v

