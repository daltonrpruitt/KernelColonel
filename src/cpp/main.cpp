#include <iostream>

#include <ConfigStore.hpp>

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        KernelColonel::ConfigStore::fromFile(argv[1]);
        std::cout << KernelColonel::ConfigStore::getConfig().dump(4) << std::endl;
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
