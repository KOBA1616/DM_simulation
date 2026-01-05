#ifndef DM_BINDINGS_TYPES_HPP
#define DM_BINDINGS_TYPES_HPP

#include "core/card_def.hpp"
#include "core/game_state.hpp" // Needed for CardInstance and Player
#include <map>
#include <vector>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

namespace dm {
    using CardDatabase = std::map<core::CardID, core::CardDefinition>;
}

// Ensure this is visible in all translation units using CardDatabase
PYBIND11_MAKE_OPAQUE(dm::CardDatabase);
PYBIND11_MAKE_OPAQUE(std::vector<dm::core::CardInstance>);
PYBIND11_MAKE_OPAQUE(std::vector<dm::core::Civilization>);
PYBIND11_MAKE_OPAQUE(std::vector<dm::core::Player>);

#endif // DM_BINDINGS_TYPES_HPP
