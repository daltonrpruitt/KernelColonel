#pragma once
#include "execution/SimpleKernelExecution.hpp"
// #include "utils/utils.hpp"

namespace KernelColonel
{

    using std::cerr;
    using std::cout;
    using std::endl;
    using std::string;
    using std::to_string;
    using std::vector;

    template<typename value_t, typename index_t>
    SimpleKernelExecution<value_t, index_t>::SimpleKernelExecution(const std::string &kernel_name_,
                                                                   const jitify::Program &program_,
                                                                   simple_check_callback_t simple_check_callback_)
        : m_kernel_name{kernel_name_}, m_program{program_}, m_simple_check_callback{simple_check_callback_} {}

    template<typename value_t, typename index_t>
    SimpleKernelExecution<value_t, index_t>::~SimpleKernelExecution()
    {
    }

    template<typename value_t, typename index_t>
    bool SimpleKernelExecution<value_t, index_t>::check(std::shared_ptr<kernel_data_t> data)
    {
        data->copyOutputFromDeviceToHost();
        const auto &host_data_vecs = data->getHostData(); 
        const auto &host_indices_vecs = data->getHostIndicies(); 
        return m_simple_check_callback(host_data_vecs[0], host_data_vecs[1], host_indices_vecs[0]);
    }

    template<typename value_t, typename index_t>
    double SimpleKernelExecution<value_t, index_t>::time_single_execution(std::shared_ptr<kernel_data_t> data, dim3 grid, dim3 block)
    {
        data->init(0);
        // using jitify::reflection::reflect;
        std::cout << "DEBUG : host data = " << data->getHostData()[0] << std::endl;

        auto launcher = m_program.kernel(m_kernel_name)
                    .instantiate<value_t, index_t>()
                    .configure(grid, block);
    
        // auto N = data->getSize();
        // auto gpu_data_ptrs = data->getDeviceDataPointers();
        // auto gpu_indices_ptrs = data->getDeviceIndiciesPointers();
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        cudaEventRecord(start);
        launcher.safe_launch(data->getSize(), data->getNamedDeviceData());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaPrintLastError();

        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return time;
        // throw std::runtime_error("KernelExecution<kernel_data_t>::time_single_execution is not currently implemented!");
    }

} // namespace KernelColonel