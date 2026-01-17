# CREST-sparse: Sparse Continuum Robot ESTimation 
Sparse nonlinear optimization solvers for various continuum robots and structures.

# Build
May need to do this to resolve dynamic link errors:

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/gtsam.conf
sudo ldconfig


setup virtual environment with pip etc.
pip install .
run test scripts
