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

    auto get_cpu_data_vector() { return super::host_data; }
    auto get_cpu_indices_vector() { return super::host_indices; }
    auto get_gpu_data_ptrs_vector() { return super::device_data_ptrs; }
    auto get_gpu_indices_ptrs_vector() { return super::device_indices_ptrs; }
    
    struct gpu_data_s
    {
        value_t* input = nullptr;
        value_t* output = nullptr;
        index_t* indices = nullptr;
    } gpu_named_data;

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
        gpu_named_data.input = device_data_ptrs[0];
        gpu_named_data.output = device_data_ptrs[1];
        gpu_named_data.indices = device_indices_ptrs[0];
    }

    int local_i = 0;
};

TEST(IKernelDataTests, Construct) {
    using KernelData_t = KernelData_Test<float, int>;
    KernelData_t data(4);
    vector<vector<KernelData_t::vt_>> cpu_data_vector = data.get_cpu_data_vector();
    vector<vector<KernelData_t::it_>> cpu_indices_vector = data.get_cpu_indices_vector();
    vector<KernelData_t::vt_* > gpu_data_ptrs_vector = data.get_gpu_data_vector();
    vector<KernelData_t::it_* > gpu_indices_ptrs_vector = data.get_gpu_indices_vector();
    
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    ASSERT_EQ(cpu_indices_vector.size(), 0);
    EXPECT_EQ(cpu_indices_vector[0].size(), 0);

    ASSERT_EQ(gpu_data_vector.size(), 2);
    EXPECT_EQ(gpu_data_vector[0], nullptr);
    EXPECT_EQ(gpu_data_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_vector.size(), 1);
    EXPECT_EQ(gpu_indices_vector[0], nullptr);
}
