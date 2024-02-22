#include <config.hpp>

using nlohmann::json;
// using nlohmann::json_schema::json_validator;

namespace KernelColonel::config 
{

nlohmann::json jsonFromFile(const std::filesystem::path &filePath)
{
    std::ifstream file(filePath);
    // TODO: expect file to exist !!!
    auto json = nlohmann::json::parse(file, nullptr, true, true);
    return json;
}


InputDataSchemaValidator::InputDataSchemaValidator(std::string name, std::string schema)
{
    // validator.set_root_schema(schema);
}

bool InputDataSchemaValidator::isValid(nlohmann::json json)
{
    // validator.validate()
    return false;
}




} // namespace KernelColonel::config
