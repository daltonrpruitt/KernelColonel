#pragma once
/**
 \* @file KernelExecution.tpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Implements KernelExecution
 *
 */

#include "execution/KernelExecution.hpp"

#include <utils/cuda_utils.hpp>

#include <algorithm>
#include <string>
#include <vector>
#include <exception>

namespace KernelColonel {

    using std::cerr;
    using std::cout;
    using std::endl;
    using std::string;
    using std::to_string;
    using std::vector;

    template <typename kernel_data_t>
    KernelExecution<kernel_data_t>::KernelExecution(/* jitify kernel?, */ std::function<bool(kernel_data_t)> check_callback_)
        : check_callback{check_callback_} {}

    template <typename kernel_data_t>
    KernelExecution<kernel_data_t>::~KernelExecution()
    {
    }

    template <typename kernel_data_t>
    bool KernelExecution<kernel_data_t>::check(std::shared_ptr<kernel_data_t> data)
    {
        return check_callback(data);
    }

    template <typename kernel_data_t>
    double KernelExecution<kernel_data_t>::time_single_execution(std::shared_ptr<kernel_data_t> data)
    {
        throw std::runtime_error("KernelExecution<kernel_data_t>::time_single_execution is not currently implemented!");
    }

} // namespace KernelColonel