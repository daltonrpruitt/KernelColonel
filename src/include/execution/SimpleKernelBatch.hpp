#pragma once
/**
 * @file SimpleKernelBatch.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Container of multiple RunSets that share the same data
 *
 */

#include <vector>
#include <memory>
#include <string>

#include "data/SimpleKernelData.hpp"
#include "execution/SimpleKernelExecution.hpp"
#include "execution/SimpleKernelRunSet.hpp"

namespace KernelColonel
{

template <typename value_t = double, typename index_t = unsigned long>
class SimpleKernelBatch
{
    using kernel_data_t = SimpleKernelData<value_t, index_t>;
    using execution_context_t = SimpleKernelExecution<value_t, index_t>;
    using run_set_t = SimpleKernelRunSet<value_t, index_t>;

  public:
    SimpleKernelBatch(std::shared_ptr<kernel_data_t> data);

    bool add_run_set(std::shared_ptr<run_set_t> run_set_ptr);
    
    bool check_all();
    bool run_all();
    bool check_all_then_run_all();

private:
    std::string m_name;
    std::shared_ptr<kernel_data_t> m_data;
    std::vector<std::shared_ptr<run_set_t>> m_run_sets;
};

} // namespace KernelColonel

#include "execution/details/SimpleKernelBatch.tpp"
