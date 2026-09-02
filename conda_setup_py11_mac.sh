#!/bin/bash
set -e

exec > >(tee -i setup_log_py11.txt) 2>&1

# get working directiry for where crest repo is located (assumes this script is run from the repo root)
CREST_SPARSE_DIR=$(pwd)
echo "Working directory: $CREST_SPARSE_DIR"
# set root dir for git repos to be one level back from the crest repo
GIT_REPOS_DIR="$CREST_SPARSE_DIR/.."
echo "Git repos directory: $GIT_REPOS_DIR"

# clean up old builds
rm -rf $CREST_SPARSE_DIR/build
rm -rf $GIT_REPOS_DIR/gtsam

# Create and activate conda environment (used instead of venv for isolation)
echo "Creating and activating conda environment 'crest'..."
# check if the environment already exists
if conda info --envs | grep -q "crest_py11"; then
    echo "Conda environment 'crest_py11' already exists. Removing it..."
    conda env remove -n crest_py11 -y
fi

echo "Conda environment 'crest_py11' does not exist. Creating it..."
conda create -n crest_py11 python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crest_py11

# Install C++ build dependencies via conda
echo "Installing C++ build dependencies via conda..."
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
conda install -c conda-forge openvdb libboost-devel tbb-devel openvdb cmake eigen pybind11 llvm-openmp "libpinocchio=4.0.0=h9a60d09_0" pinocchio=4.0.0 openssh git -y
# conda install -c conda-forge openvdb libboost-devteel cmake eigen pybind11 llvm-openmp -y

# Build/Install GTSAM (into conda prefix so it stays isolated from system)
echo "Cloning and building GTSAM..."
cd $GIT_REPOS_DIR
# Forked GTSAM 4.3a1 carrying our constrained-module heap-overflow fix
# (NonlinearEquality/InequalityConstraint::violationVector: middleCols -> segment).
# git clone https://github.com/joeliechty/gtsam.git   # HTTPS alternative (public fork)
git clone git@github.com:joeliechty/gtsam.git
cd gtsam
git checkout release-4.3a1-fixes  # 4.3a1 + constrained-module fixes
mkdir build
cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF
make -j8
make install

# Make GTSAM libraries visible to the dynamic linker within the conda env
echo "Setting DYLD_LIBRARY_PATH for GTSAM libraries..."
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Build/Install CREST-sparse
echo "Building and installing CREST-sparse..."
cd $CREST_SPARSE_DIR
# Runtime dependencies are declared in pyproject.toml; installing the package
# brings them in. [viz,web] adds the PyVista windows the demo scripts open and
# the viser workbench.
pip install ".[viz,web]" -v

# pip's numpy/scipy wheels are built against macOS Accelerate which may lack newer
# LAPACK symbols on this OS version. Replace them with conda versions (use openblas).
#
# This MUST run AFTER `pip install`, not before it: numpy and scipy are declared
# dependencies now, so installing the package resolves them, and doing the swap
# first would simply have the Accelerate wheels put back. They are declared with
# a lower bound rather than an exact pin for the same reason -- conda's newer
# build satisfies it, so a later `pip install -e .` will not undo this either.
pip uninstall numpy scipy -y
conda install -c conda-forge numpy scipy --force-reinstall -y
