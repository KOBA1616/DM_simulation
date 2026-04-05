#include "bindings/bindings.hpp"
#include "bindings/types.hpp"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>

void bind_common(py::module& m) {
    // 再発防止: 非空 CardDatabase を Python へ返す際に shared_ptr holder 経路で
    // access violation が発生するケースがあるため、bind_map は既定 holder を使う。
    py::bind_map<dm::CardDatabase>(m, "CardDatabase");
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
    // NOTE: 再発防止 — IS_NATIVE を True に設定してネイティブモジュールであることをテストから判別できるようにする。
    // テストが Python フォールバックとネイティブを区別するときに参照する。
    m.attr("IS_NATIVE") = true;
    // Ensure core types (CardDefinition, EffectDef, etc.) are bound
    // before binding container/map helpers so value conversions use
    // the proper pybind11 type registrations.
    bind_core(m);
    bind_common(m);
    bind_engine(m);
    bind_ai(m);
    // Native -> Python event bridge
    // NOTE: Temporarily disabled to avoid linker-time dependency while
    // we iterate on loader fixes. Re-enable once symbol resolution is
    // verified or the bridge is moved to a separate compilation unit.
    // try { dm::bindings::bind_event_bridge(m); } catch(...) {}
    // Inference bindings (ONNX / LibTorch wrappers)
    try { bind_inference(m); } catch(...) {}
    try {
        std::ofstream out("logs/module_init.txt", std::ios::app);
        if (out) {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            out << "MODULE_INIT end pid=" << std::this_thread::get_id() << " time=" << now << "\n";
            out.close();
        }
    } catch(...) {}
}
