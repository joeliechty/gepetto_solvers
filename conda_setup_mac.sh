#!/bin/bash
set -e

exec > >(tee -i setup_log.txt) 2>&1

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
if conda info --envs | grep -q "crest"; then
    echo "Conda environment 'crest' already exists. Removing it..."
    conda env remove -n crest -y
fi

echo "Conda environment 'crest' does not exist. Creating it..."
conda create -n crest python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crest

# Install C++ build dependencies via conda
echo "Installing C++ build dependencies via conda..."
conda install -c conda-forge openvdb libboost-devel tbb-devel cmake eigen pybind11 llvm-openmp -y
# conda install -c conda-forge openvdb libboost-devteel cmake eigen pybind11 llvm-openmp -y

# Build/Install GTSAM (into conda prefix so it stays isolated from system)
echo "Cloning and building GTSAM..."
cd $GIT_REPOS_DIR
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1  # Tested GTSAM version
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
pip install -r requirements.txt
# pip's numpy/scipy wheels are built against macOS Accelerate which may lack newer
# LAPACK symbols on this OS version. Replace them with conda versions (use openblas).
pip uninstall numpy scipy -y
conda install -c conda-forge numpy scipy --force-reinstall -y
pip install . -v
