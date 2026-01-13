#include "bindings.h"


PYBIND11_MODULE(crest_sparse, m) {
    bind_cosserat_rod(m);
    bind_cosserat_dynamics(m);
    bind_cosserat_shell(m);
    bind_parallel_robot(m);
    bind_tendon_robot(m);
}