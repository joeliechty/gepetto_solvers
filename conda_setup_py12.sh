#!/bin/bash


exec > >(tee -i setup_log_py12.txt) 2>&1

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
echo "Creating and activating conda environment 'crest_py12'..."
# check if the environment already exists
if conda info --envs | grep -q "crest_py12"; then
    echo "Conda environment 'crest_py12' already exists. Removing it..."
    conda env remove -n crest_py12 -y
fi

echo "Conda environment 'crest_py12' does not exist. Creating it..."
conda create -n crest_py12 python=3.12 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crest_py12

# Install C++ build dependencies via conda
echo "Installing C++ build dependencies via conda..."
# conda install -c conda-forge cmake eigen pybind11 boost libgomp openvdb -y
# conda install -c conda-forge cmake eigen pybind11 boost libgomp openvdb tbb-devel gcc_linux-64 gxx_linux-64 -y
conda install -c conda-forge cmake eigen pybind11 boost libgomp openvdb tbb-devel suitesparse gcc_linux-64 gxx_linux-64 openssh git -y

# Build/Install GTSAM (into conda prefix so it stays isolated from system)
echo "Cloning and building GTSAM..."
cd $GIT_REPOS_DIR
# Forked GTSAM 4.3a1 carrying our constrained-module heap-overflow fix
# (NonlinearEquality/InequalityConstraint::violationVector: middleCols -> segment).
git clone https://github.com/joeliechty/gtsam.git
# git clone git@github.com:joeliechty/gtsam.git   # SSH alternative
cd gtsam
git checkout release-4.3a1-fixes  # 4.3a1 + constrained-module fixes
mkdir build
cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
# cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF -DGTSAM_BUILD_TESTS=OFF -DGTSAM_WITH_TBB=ON
make -j8
make install

# Make GTSAM libraries visible to the dynamic linker within the conda env
echo "Setting LD_LIBRARY_PATH for GTSAM libraries..."
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Build/Install CREST-sparse
echo "Building and installing CREST-sparse..."
cd $CREST_SPARSE_DIR
pip install -r requirements.txt
pip install . -v

