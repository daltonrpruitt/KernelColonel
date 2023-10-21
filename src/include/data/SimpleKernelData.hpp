/**
 * @file SimpleKernelData.cuh
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Simplest IKernelData partial specialization
 * @version 0.1
 * @date 2023-04-06
 * 
 * @copyright Copyright (c) 2023
 * 
 * NOTE: I am very unsure if this pattern (partial specialization subclass?) is long-term 
 *  what needs to happen; it just seemed useful in the short term when implementing my simplified goals in 
 *  `simeple_reorganization_structure.cu` the week of 2023-04-03.
 * 
 */
#pragma once

#include "data/IKernelData.hpp"

#include <functional>

namespace KernelColonel 
{

template<typename value_t, typename index_t> 
struct SimpleKernelData_gpu_data_s
{
    value_t* input = nullptr;
    value_t* output = nullptr;
    index_t* indices = nullptr;
};

template<typename value_t = double, typename index_t = unsigned long> 
class SimpleKernelData : public IKernelData<value_t, index_t, 1, 1, 1, SimpleKernelData_gpu_data_s<value_t,index_t>>
{
  public:
    using vt_ = value_t;
    using it_ = index_t;
    using super = IKernelData<vt_, it_, 1, 1, 1, SimpleKernelData_gpu_data_s<vt_,it_>>;
    using super::N;
    using super::host_data;
    using super::host_indices;
    using super::device_data_ptrs;
    using super::device_indices_ptrs;
    using super::gpu_named_data;
    using init_input_func_t = std::function<std::vector<vt_>(unsigned int)>;
    using init_indices_func_t = std::function<std::vector<it_>(unsigned int)>;
    using check_outputs_func_t = std::function<bool(unsigned long long, vt_*, vt_*)>;

    SimpleKernelData(unsigned long long n) : super(n) {}
    ~SimpleKernelData() = default;

    void setDataSizes(unsigned long long input_size_, unsigned long long output_size_, unsigned long long indices_size_)
    {
        super::input_size=input_size_;
        super::output_size=output_size_;
        super::indices_size=indices_size_;
    }

    void setInitInputsFunc(init_input_func_t init_inputs_) { m_init_inputs = init_inputs_; }
    void setInitIndicesFunc(init_indices_func_t init_indices_) { m_init_indices = init_indices_; }
    void setCheckOutputsFunc(check_outputs_func_t check_outputs_) { m_check_outputs = check_outputs_; }
    
  private:
    void initInputsCpu() override
    { 
        if(!m_init_inputs) {
            throw std::runtime_error("Method to initialize inputs has not been set!");
        } else {
            host_data[0] = m_init_inputs(N);
            host_data[1].resize(N);
        }
    }
    
    void initIndicesCpu() override
    { 
        if(!m_init_indices) {
            std::cerr << "Method to initialize indices has not been set!\n";
            std::cerr << "\tSetting all indices to 0!" << std::endl;
            super::indices_size = 0;
            host_indices[0] = std::vector<it_>(0, super::indices_size);
        } else {
            host_indices[0] = m_init_indices(N);
        }
    }
    
    bool checkOutputsCpu()
    { 
        if(!m_check_outputs) {
            throw std::runtime_error("Method to check outputs has not been set!");
        } else {
            return m_check_outputs(N, device_data_ptrs[0], device_data_ptrs[1]);
        }
    }

    void setGpuNamedData() override 
    {
        gpu_named_data.input = device_data_ptrs[0];
        gpu_named_data.output = device_data_ptrs[1];
        gpu_named_data.indices = device_indices_ptrs[0];
    }
    
    init_input_func_t m_init_inputs;
    init_indices_func_t m_init_indices;
    check_outputs_func_t m_check_outputs;
};

} // namespace KernelColonel
