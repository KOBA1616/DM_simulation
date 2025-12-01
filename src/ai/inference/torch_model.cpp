#ifdef USE_LIBTORCH
#include "torch_model.hpp"
#include <iostream>

using namespace dm::ai;

bool TorchModel::load(const std::string& path) {
    try {
        module = torch::jit::load(path);
        module.eval();
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return false;
    }
}

std::pair<std::vector<float>, std::vector<float>> TorchModel::predict(const std::vector<float>& input) {
    // Convert input to tensor
    torch::Tensor in = torch::from_blob((float*)input.data(), {(long)1, (long)input.size()});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(in);
    auto output = module.forward(inputs).toTuple();
    // Expecting tuple (policy_tensor, value_tensor)
    auto policy_t = output->elements()[0].toTensor();
    auto value_t = output->elements()[1].toTensor();
    std::vector<float> policy(policy_t.data_ptr<float>(), policy_t.data_ptr<float>() + policy_t.numel());
    std::vector<float> value(value_t.data_ptr<float>(), value_t.data_ptr<float>() + value_t.numel());
    return {policy, value};
}
#endif
