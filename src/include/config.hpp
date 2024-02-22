#pragma once

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <nlohmann/json.hpp>
#include <nlohmann/json-schema.hpp>

namespace KernelColonel::config
{

    nlohmann::json jsonFromFile(const std::filesystem::path &filePath);

    class InputDataSchemaValidatorsCollection
    {
    public:
        InputDataSchemaValidatorsCollection(std::string name, std::string schema);
        // void validate();

    private:
        // nlohmann::json_validator validator;
    };

    class InputDataSchemaValidator
    {
    public:
        InputDataSchemaValidator(std::string name, std::string schema);
        bool isValid(nlohmann::json json);

    private:
        // nlohmann::json_validator validator;
    };

} // namespace KernelColonel::config
