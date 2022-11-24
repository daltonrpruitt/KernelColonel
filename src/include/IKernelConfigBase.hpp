#pragma once 
/**
 * @file IKernelInputs.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides a wrapper surrounding the GPU-specific data pointers and other inputs
 * 
 */

#include <vector>
#include <string>

namespace KernelColonel {

/**
 * @brief Wrapper for GPU device pointers and other kernel inputs (i.e. used to reduce number of kernel inputs)
 * 
 * TODO: Possibly will rework to make a variadic template
 * 
 * @tparam vt Value type (data arrays)s
 * @tparam it Index type (indirection arrays)
 */
template<typename vt, typename it>
struct KernelInputs { 
    KernelInputs() = default;
    virtual ~KernelInputs() = 0;
        
    int gpu_device_id;
    unsigned long long input_scale=0;
};

} // namespace KernelColonel
