#ifndef DM_BINDINGS_TYPES_HPP
#define DM_BINDINGS_TYPES_HPP

#include "core/card_def.hpp"
#include <map>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace dm {
    using CardDatabase = std::map<core::CardID, core::CardDefinition>;
}

// Ensure this is visible in all translation units using CardDatabase
PYBIND11_MAKE_OPAQUE(dm::CardDatabase);

#endif // DM_BINDINGS_TYPES_HPP
