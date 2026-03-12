#!/bin/bash

exec > >(tee -i setup_log.txt) 2>&1

conda create -n crest python=3.11 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate crest
conda install -c conda-forge cmake eigen pybind11 boost numpy scipy matplotlib -y

# install gtsam
cd ~/git_repos
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.3a1  # match the version you need
mkdir build && cd build
cmake .. \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -DGTSAM_BUILD_PYTHON=OFF \
  -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF \
  -DGTSAM_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

cd ~/git_repos/crest-sparse

pip install -r requirements.txt
pip install .

