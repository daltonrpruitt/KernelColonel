#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>

namespace KernelColonel {

struct ConfigStore {
  private:
    nlohmann::json config;

  public:
    static ConfigStore& instance() {
        static ConfigStore instance; 
        return instance;
    }

    static const nlohmann::json& getConfig() {
        return instance().config; 
    }

    static void fromFile(const std::filesystem::path& filePath);

    template<typename value_t>
    value_t operator[](std::string key) {
        return instance().config[key];
    }

    template<typename value_t>
    value_t at(std::string key) {
        return instance().config.at(key);
    }

private:
    ConfigStore(){};
    // ConfigStore(const ConfigStore&);
    // ConfigStore& operator=(const ConfigStore&);
};

} // namespace KernelColonel 
