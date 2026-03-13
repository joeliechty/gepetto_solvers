#!/bin/bash
set -e

exec > >(tee -i setup_log.txt) 2>&1

# Create and activate conda environment (used instead of venv for isolation)
echo "Creating and activating conda environment 'crest'..."
conda create -n crest python=3.12 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crest

# Install C++ build dependencies via conda
echo "Installing C++ build dependencies via conda..."
conda install -c conda-forge cmake eigen pybind11 boost boost-cpp -y

# Build/Install GTSAM (into conda prefix so it stays isolated from system)
echo "Cloning and building GTSAM..."
cd ~/git_repos
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1  # Tested GTSAM version
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j8
make install

# Make GTSAM libraries visible to the dynamic linker within the conda env
echo "Setting DYLD_LIBRARY_PATH for GTSAM libraries..."
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# Build/Install CREST-sparse
echo "Building and installing CREST-sparse..."
cd ~/git_repos/crest-sparse
pip install -r requirements.txt
# pip's numpy/scipy wheels are built against macOS Accelerate which may lack newer
# LAPACK symbols on this OS version. Replace them with conda versions (use openblas).
pip uninstall numpy scipy -y
conda install -c conda-forge numpy scipy --force-reinstall -y
pip install . -v
