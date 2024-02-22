#include <iostream>

#include <config.hpp>

int main(int argc, char **argv) {
    if(argc != 2)
    {
        std::cout << "Must provide a configuration file! : `kernel_colonel_exec config.json`" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        auto config = KernelColonel::config::jsonFromFile(argv[1]);

        for(const auto& entry : config)
        {
            std::cout << "Entry = " << entry.dump(4) << std::endl;
        }
        
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
