#pragma once

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

#include <vector>
#include <string>

namespace dm::ai {

class TorchModel {
public:
#ifdef USE_LIBTORCH
    TorchModel() = default;
    bool load(const std::string& path);
    std::pair<std::vector<float>, std::vector<float>> predict(const std::vector<float>& input);
#else
    // Stubs when LibTorch not enabled
    TorchModel() {}
    bool load([[maybe_unused]] const std::string&) { return false; }
    std::pair<std::vector<float>, std::vector<float>> predict([[maybe_unused]] const std::vector<float>&) { return { {}, {} }; }
#endif
};

} // namespace dm::ai
