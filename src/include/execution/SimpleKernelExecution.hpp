#pragma once 
/**
 \* @file SimpleKernelExecution.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Specialization of IKernelExecution that uses SimpleKernelData
 * 
 */

#include <vector>
#include <memory>
#include <string>

#include "data/SimpleKernelData.hpp"

namespace KernelColonel {

/**
 * @brief Container for GPU kernel execution configuration
 * 
 * TODO: Determine if should have be templated on all 
 * 
 * Necessary because different kernel contexts may use the same input data, but will the data
 * will be removed and re-computed and copied between separate kernel context instance executions.
 * 
 * @tparam value_t Value type (data arrays)
 * @tparam it Index type (indirection arrays)
 */
template<typename value_t, typename index_t>
class SimpleKernelExecution : public IKernelExecution<SimpleKernelData<value_t, index_t>>
{ 
  public:
    SimpleKernelExecution(unsigned long long n);
    ~SimpleKernelExecution();

};

} // namespace KernelColonel
