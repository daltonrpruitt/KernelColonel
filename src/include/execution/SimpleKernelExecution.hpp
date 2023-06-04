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
template<typename value_t = double, typename index_t = unsigned long>
class SimpleKernelExecution
{
  public:
    using kernel_data_t = SimpleKernelData<value_t, index_t>;
    using simple_check_callback_t = std::function<bool(const std::vector<value_t>&, 
                                                       const std::vector<value_t>&, 
                                                       const std::vector<index_t>&)>;

    SimpleKernelExecution(const std::string &name_, 
                          /* jitify stuff, */
                          simple_check_callback_t simple_check_callback_);
    ~SimpleKernelExecution();

    bool check(std::shared_ptr<kernel_data_t> data);

    double time_single_execution(std::shared_ptr<kernel_data_t> data);

  private:
    std::string name;
    simple_check_callback_t simple_check_callback;

};

} // namespace KernelColonel

#include <execution/details/SimpleKernelExecution.tpp>
