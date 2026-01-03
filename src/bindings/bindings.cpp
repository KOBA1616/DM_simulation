#include "bindings/bindings.hpp"
#include "bindings/types.hpp"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

void bind_common(py::module& m) {
    py::bind_map<dm::CardDatabase, std::shared_ptr<dm::CardDatabase>>(m, "CardDatabase");
}

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Module";
    bind_common(m);
    bind_core(m);
    bind_engine(m);
    bind_ai(m);
}
