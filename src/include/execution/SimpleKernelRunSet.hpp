#pragma once
/**
 * @file KernelRunSet.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Encapsulation of multiple runs of a given kernel on given input data
 *
 */

#include <vector>
#include <memory>
#include <string>

#include "data/SimpleKernelData.hpp"
#include "execution/SimpleKernelExecution.hpp"

namespace KernelColonel
{

template <typename value_t = double, typename index_t = unsigned long>
class SimpleKernelRunSet
{
    using kernel_data_t = SimpleKernelData<value_t, index_t>;
    using execution_context_t = SimpleKernelExecution<value_t, index_t>;

  public:
    SimpleKernelRunSet(std::shared_ptr<execution_context_t> exec_ctx,
                        std::shared_ptr<kernel_data_t> data,
                        dim3 grid_size = 1,
                        dim3 block_size = 1,
                        unsigned int num_runs = 25);
    bool check();
    double run_single();
    bool run_all();
    bool check_and_run_all();

    bool check_data_pointer_is_same(std::shared_ptr<kernel_data_t> data);

    std::vector<double> get_run_timings() { return m_run_timings; }

private:

    std::string m_name;
    std::shared_ptr<execution_context_t> m_execution_ctx;
    std::shared_ptr<kernel_data_t> m_data;
    dim3 m_grid_size;
    dim3 m_block_size;
    unsigned int m_num_runs;
    std::vector<double> m_run_timings;
    bool m_checked;
};

} // namespace KernelColonel

#include "execution/details/SimpleKernelRunSet.tpp"
