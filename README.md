# CREST-sparse: Sparse Continuum Robot ESTimation 
Sparse nonlinear optimization solvers for various continuum robots and structures.

# Build

Clone repository
```bash
git clone https://github.com/fergujm2/crest-sparse.git
cd crest-sparse
```

Setup venv
```bash
python3 -m venv .venv
source .venv/bin/activate
```

TODO install python stuff
TODO install gtsam stuff

Build with 
```bash
pip install .
```

May need to do this to resolve dynamic link errors:

```bash
echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/gtsam.conf
sudo ldconfig
```

TODO run test scripts with plotting