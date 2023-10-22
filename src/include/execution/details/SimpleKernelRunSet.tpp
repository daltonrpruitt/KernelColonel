#pragma once 
/**
 * @file KernelRunSet.hpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Encapsulation of multiple runs of a given kernel on given input data
 * 
 */

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include "data/SimpleKernelData.hpp"
#include "execution/SimpleKernelRunSet.hpp"


namespace KernelColonel {

template<typename value_t, typename index_t>
SimpleKernelRunSet<value_t, index_t>::SimpleKernelRunSet(std::shared_ptr<execution_context_t> exec_ctx, 
               std::shared_ptr<kernel_data_t> data,
               dim3 grid_size, 
               dim3 block_size,
               unsigned int num_runs)
    : m_execution_ctx(exec_ctx), 
      m_data(data), 
      m_grid_size(grid_size), 
      m_block_size(block_size), 
      m_num_runs(num_runs), 
      m_checked(false)
    {}

template<typename value_t, typename index_t>
bool SimpleKernelRunSet<value_t, index_t>::check()
{
    if(m_checked)
    {
        return m_checked;
    }
    
    m_execution_ctx->time_single_execution(m_data, m_grid_size, m_block_size);
    if(m_execution_ctx->check(m_data))
    {
        m_checked = true;
    }
    return m_checked;
}

template<typename value_t, typename index_t>
double SimpleKernelRunSet<value_t, index_t>::run_single()
{
    return m_execution_ctx->time_single_execution(m_data, m_grid_size, m_block_size);
}

template<typename value_t, typename index_t>
bool SimpleKernelRunSet<value_t, index_t>::run_all()
{
    if(m_run_timings.size() != 0)
    {
        std::cerr << "SimpleKernelRunSet '" << m_name << "' : Cannot rerun a SimpleKernelRunSet!" << std::endl;
        return false;
    }

    for(unsigned int i=0; i < m_num_runs; ++i)
    {
        auto time = run_single();
        if(time < 0)
        {
            std::cerr << "SimpleKernelRunSet '" << m_name << "' : Failed run #" << i <<" !" << std::endl;
            return false;
        }
        m_run_timings.push_back(time);
    }
    return true;
}

template<typename value_t, typename index_t>
bool SimpleKernelRunSet<value_t, index_t>::check_and_run_all()
{
    if(!m_checked && !check()) 
    {
        return false;
    }
    run_all();
    return true;
}

} // namespace KernelColonel
