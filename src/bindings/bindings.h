#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

void bind_cosserat_rod(pybind11::module& m);
void bind_cosserat_dynamics(pybind11::module& m);
void bind_digits_tendon(pybind11::module& m);
void bind_hand(pybind11::module& m);