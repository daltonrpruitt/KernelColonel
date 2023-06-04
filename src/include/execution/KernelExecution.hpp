#pragma once 
/**
 \* @file KernelExecution.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides a wrapper surrounding the data inputs/outputs and indices 
 *          GPU kernel execution (in IKernelContext)
 * 
 */

#include <vector>
#include <memory>
#include <string>

#include "data/IKernelData.hpp"

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
template<typename kernel_data_t>
class KernelExecution { 
  public:
    using check_callback_t = std::function<bool(kernel_data_t)>;

    KernelExecution(/* jitify stuff, */
        check_callback_t check_callback_);
    ~KernelExecution();

    /**
     * @brief Run kernel on data and check result against CPU-computed result
     * 
     * Ensure KernelData is initialized, execute kernel on GPU data, check result against "correct" values.
     * TODO: Could have "correct" results computed locally ahead of time and stored with the data structure?
     *        Or could just be recomputed everytime.... I think it makes more sense to store with the data structure, 
     *        but should it be computed by the data structure??? NO, because the data structure knows nothing about 
     *        the inputs/outputs/indices other than the type and number of each (and length). 
     *        So, we could compute the output correctness here (in the derived class), then store with the data
     *        structure, assume the output param is the first param of the kernel, then compute against the correct
     *        version after execution.
     * 
     * NOTE: The best practice when checking for correctness is probably to completely zero out the output before 
     * the next execution. This function should do that after checking, but the regular execute will not, since it
     * assumes correctness and so can avoid that overhead. 
     * 
     * NOTE: This does not ensure correctness of the CPU and GPU algorithms supplied by the user, only that both 
     * algorithms compute the same result! 
     * 
     * @return true The output of the kernel matches the CPU-computed "correct" output 
     * @return false Failed to initialize properly (handling taken care of by owner of object)
     */
    bool check(std::shared_ptr<kernel_data_t> data);

    /**
     * @brief Execute a run once and give the GPU execution time
     * 
     * @return double Execution time of a single run (GPU time only) 
     */
    double time_single_execution(std::shared_ptr<kernel_data_t> data);

  private: 
    std::string name;
    check_callback_t check_callback;

};

} // namespace KernelColonel

#include "details/KernelExecution.tpp"
