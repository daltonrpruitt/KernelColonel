#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        std::ifstream file(argv[1]);
        auto config = nlohmann::json::parse(file, nullptr, true, true);
        std::cout << config.dump(4) << std::endl;
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
