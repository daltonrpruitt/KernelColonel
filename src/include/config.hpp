#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>

namespace KernelColonel::config
{
    extern nlohmann::json config;
    void fromFile(const std::filesystem::path &filePath);

    namespace required_fields
    {
        extern std::string config_name;
    } // namespace required_fields

    namespace optional_fields
    {
        extern std::string log_level;
    } // namespace required_fields

} // namespace KernelColonel::config
