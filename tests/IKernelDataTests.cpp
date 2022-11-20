#include <gtest/gtest.h>

#include "IKernelData.cuh"


// Add some tests for IKernelData class !!!
// Can derive from IKernelData and perform some operations in lambdas???

template<typename value_t, typename index_t> 
class KernelData_Test : public IKernelData<value_t, index_t, 1, 1, 1>
{
  public:
    using super = IKernelData<value_t, index_t, 1, 1, 1>;
    using vt_ = value_t;
    using it_ = index_t;
    
    KernelData_Test(unsigned long long n) : super(n) {}
    ~KernelData_Test() = default;

    void set_extra_params(int i)
    {
        local_i = i;
    }

    auto get_cpu_data() { return super::host_data; }
    auto get_gpu_data() { return super::device_data_ptrs; }
    
    struct
    {
        value_t* input = nullptr;
        value_t* output = nullptr;
        index_t* indices = nullptr;
    } gpu_data;

  private:
    void init_inputs_cpu() override 
    {
        for(int i=0; i< N; ++i)
        {
            host_data[0].push_back(static_cast<value_t>(i));
        }
    }

    void init_indices_cpu() override 
    {
        for(int i=0; i<N; ++i)
        {
            host_indices[0].push_back(static_cast<index_t>(i));
        }        
    }

    void set_gpu_named_data() override 
    {
        gpu_data.input = device_data_ptrs[0];
        gpu_data.output = device_data_ptrs[1];
        gpu_data.indices = device_indices_ptrs[0];
    } gpu_named_data;

    int local_i = 0;
};

TEST(IKernelDataTests, Construct) {
    using KernelData_t = KernelData_Test<float, int>;
    KernelData_t data(4);
    vector<vector<KernelData_t::vt_>> cpu_data = data.get_cpu_data();
    vector<KernelData_t::vt_* > gpu_data = data.get_gpu_data();
    
    ASSERT_EQ(cpu_data.size(), 2);
    ASSERT_EQ(cpu_data[0].size(), 0);
    ASSERT_EQ(cpu_data[1].size(), 0);

    ASSERT_EQ(gpu_data.size(), 2);
    ASSERT_EQ(gpu_data[0], nullptr);
    ASSERT_EQ(gpu_data[1], nullptr);
}
