#include "bindings/bindings.hpp"

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Module";
    bind_core(m);
    bind_engine(m);
    bind_ai(m);
}
