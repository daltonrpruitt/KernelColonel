#pragma once
#include "execution/SimpleKernelExecution.hpp"

namespace KernelColonel
{

    using std::cerr;
    using std::cout;
    using std::endl;
    using std::string;
    using std::to_string;
    using std::vector;

    template<typename value_t, typename index_t>
    SimpleKernelExecution<value_t, index_t>::SimpleKernelExecution(const std::string &name_,
                            /* jitify kernel?, */ 
                            simple_check_callback_t simple_check_callback_)
        : simple_check_callback{simple_check_callback_} {
            this->name = name_;
        }

    template<typename value_t, typename index_t>
    SimpleKernelExecution<value_t, index_t>::~SimpleKernelExecution()
    {
    }

    template<typename value_t, typename index_t>
    bool SimpleKernelExecution<value_t, index_t>::check(std::shared_ptr<kernel_data_t> data)
    {
        const auto &host_data_vecs = data->getHostData(); 
        const auto &host_indices_vecs = data->getHostIndicies(); 
        return simple_check_callback(host_data_vecs[0], host_data_vecs[1], host_indices_vecs[0]);
    }

    template<typename value_t, typename index_t>
    double SimpleKernelExecution<value_t, index_t>::time_single_execution(std::shared_ptr<kernel_data_t> data)
    {
        throw std::runtime_error("KernelExecution<kernel_data_t>::time_single_execution is not currently implemented!");
    }

} // namespace KernelColonel