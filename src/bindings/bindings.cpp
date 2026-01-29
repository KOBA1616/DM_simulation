#include "bindings/bindings.hpp"
#include "bindings/types.hpp"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>

void bind_common(py::module& m) {
    py::bind_map<dm::CardDatabase, std::shared_ptr<dm::CardDatabase>>(m, "CardDatabase");
}

PYBIND11_MODULE(dm_ai_module, m) {
    // Early init diagnostic: write a short file to help bisect importer crashes
    try {
        std::filesystem::create_directories("logs");
        std::ofstream out("logs/module_init.txt", std::ios::app);
        if (out) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            out << "MODULE_INIT start pid=" << std::this_thread::get_id() << " time=" << now << "\n";
            out.close();
        }
    } catch(...) {}
    m.doc() = "Duel Masters AI Module";
    bind_common(m);
    bind_core(m);
    bind_engine(m);
    bind_ai(m);
    try {
        std::ofstream out("logs/module_init.txt", std::ios::app);
        if (out) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            out << "MODULE_INIT end pid=" << std::this_thread::get_id() << " time=" << now << "\n";
            out.close();
        }
    } catch(...) {}
}
