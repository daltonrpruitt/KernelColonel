#include <iostream>

#include <config.hpp>

int main(int argc, char **argv) {
    assert(argc == 2);
    try {
        KernelColonel::config::fromFile(argv[1]);
        std::cout << KernelColonel::config::config.dump(4) << std::endl;
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
