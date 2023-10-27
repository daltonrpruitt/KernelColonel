#pragma once 
/**
 * @file SimpleKernelBatcht.tpp
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * 
 */

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include "data/SimpleKernelData.hpp"
#include "execution/SimpleKernelBatch.hpp"


namespace KernelColonel {

template<typename value_t, typename index_t>
SimpleKernelBatch<value_t, index_t>::SimpleKernelBatch(std::shared_ptr<kernel_data_t> data)
    : m_data(data)
    {}

template<typename value_t, typename index_t>
bool SimpleKernelBatch<value_t, index_t>::add_run_set(std::shared_ptr<run_set_t> run_set_ptr)
{
    if(!run_set_ptr->check_data_pointer_is_same(m_data))
    {
        return false;
    }
    m_run_sets.push_back(run_set_ptr);
    return true;
}


template<typename value_t, typename index_t>
bool SimpleKernelBatch<value_t, index_t>::check_all()
{
    bool pass = true;
    for(auto& run_set : m_run_sets)
    {
        if(!run_set->check())
        {
            pass = false;
        }
    }
    return pass;
}

template<typename value_t, typename index_t>
bool SimpleKernelBatch<value_t, index_t>::run_all()
{

    bool pass = true;
    for(auto& run_set : m_run_sets)
    {
        if(!run_set->run_all())
        {
            pass = false;
        }
    }
    return pass;
}

template<typename value_t, typename index_t>
bool SimpleKernelBatch<value_t, index_t>::check_all_then_run_all()
{
    if(!check_all()) 
    {
        return false;
    }
    return run_all();
}

} // namespace KernelColonel
