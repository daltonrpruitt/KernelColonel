#include <ConfigStore.hpp>

namespace KernelColonel {

void ConfigStore::fromFile(const std::filesystem::path& filePath) {
    std::ifstream file(filePath);
    auto new_config = nlohmann::json::parse(file, nullptr, true, true);
    if(instance().config.is_null()) {
        instance().config = new_config;
    } else {
        instance().config.insert(new_config.begin(), new_config.end());
    }
}

} // namespace KernelColonel 
