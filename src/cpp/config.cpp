#include <config.hpp>

namespace KernelColonel
{

    namespace config
    {

        nlohmann::json config = nlohmann::json::parse(R"(
{
"Config Name":"default"
}
)", 
                                                      nullptr, true, true);

        void fromFile(const std::filesystem::path &filePath)
        {
            std::ifstream file(filePath);
            auto new_config = nlohmann::json::parse(file, nullptr, true, true);
            if (!new_config.count(required_fields::config_name)) 
            {
                std::cout << "Missing field in " << filePath << " : '" << required_fields::config_name << "'" << std::endl;
            }

            if (config.is_null())
            {
                config = new_config;
            }
            else
            {
                config.insert(new_config.begin(), new_config.end());
            }
        }
        namespace required_fields
        {
            std::string config_name = "Config Name";
        } // namespace required_fields

        namespace optional_fields
        {
            std::string log_level = "Log Level";
        } // namespace required_fields

    } // namespace config

} // namespace KernelColonel
