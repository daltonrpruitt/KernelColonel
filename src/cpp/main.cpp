#include <iostream>

#include <config.hpp>
#include <execution/SimpleKernelBatch.hpp>

int main(int argc, char **argv) {
    if(argc != 2)
    {
        std::cout << "Must provide a configuration file! : `kernel_colonel_exec config.json`" << std::endl;
        return EXIT_FAILURE;
    }

    try {
        auto config = KernelColonel::config::jsonFromFile(argv[1]);

        // TODO: validate all config items

        std::vector<KernelColonel::SimpleKernelRunSet> run_sets;
        std::map<KernelColonel::SimpleKernelData> input_data_instances;

        // process config items in loop
        for(const auto& entry : config)
        {
            // make new run (or set of runs) for execution context config
                // TODO: should handle processing of ranges of various configurables...
                // For now, single value settings; no ranges
            std::cout << "Entry = " << entry.dump(4) << std::endl;
            unsigned long auto_grid_size = static_cast<unsigned long>(entry["input_size"]) / static_cast<unsigned long>(entry["block_size"]);

            if (entry["grid_size"] == R"("auto")"_json) 
            {
                std::cout << "grid_size is auto : ==> grid_size=" << auto_grid_size << std::endl;
            }
            // run_sets.append(KernelColonel::SimpleRu)
            // need global list of registered data generation functions?

            // if input configuration not present in "global" inputs list, 
                // construct new one and add to global list
            // Use handle to input data in run set
        }

        // Sort runs into buckets with same data, then by same kernel
        // For each batch (by input data):
            // for each run set:
                // initialize the data (copy from CPU to GPU memory)
                // Run each kernel run set once
                // if output not valid, 
                    // report error, skip this kernel config
                // Rerun (with fresh L2 cache?) X times, record data
        // provide summary of errors?

    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
