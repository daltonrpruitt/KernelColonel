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

#include "data/IKernelData.hpp"
namespace KernelColonel 
{

template<typename value_t, typename index_t> 
struct SimpleKernelData_gpu_data_s
{
    value_t* input = nullptr;
    value_t* output = nullptr;
    index_t* indices = nullptr;
};

template<typename value_t, typename index_t> 
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

    SimpleKernelData(unsigned long long n) : super(n) {}
    ~SimpleKernelData();

    using void_func_t = std::function<void(void)>;
    void setInitInputFunc(void_func_t init_inputs_) { m_init_inputs = init_inputs_};
    void setInitIndicesFunc(void_func_t init_indices_) { m_init_indices = init_indices_ };
    
  private:
    void initInputsCpu() override;
    void initIndicesCpu() override;
    void setGpuNamedData() override;

    void setGpuNamedData() override 
    {
        gpu_named_data.input = device_data_ptrs[0];
        gpu_named_data.output = device_data_ptrs[1];
        gpu_named_data.indices = device_indices_ptrs[0];
    }
    
    std::function<void(void)> m_init_inputs;
    std::function<void(void)> m_init_indices;
};

} // namespace KernelColonel
