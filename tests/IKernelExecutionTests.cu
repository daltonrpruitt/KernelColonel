#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "IKernelExecution.hpp"
using namespace KernelColonel;

template<typename value_t, typename index_t> 
struct gpu_data_s
{
    value_t* input = nullptr;
    value_t* output = nullptr;
    index_t* indices = nullptr;
};

/*
template<typename value_t, typename index_t> 
class IKernelExecution_Test : public IIKernelExecution<value_t, index_t, 1, 1, 1, gpu_data_s<value_t,index_t>>
{
  public:
    using vt_ = value_t;
    using it_ = index_t;
    using super = IIKernelExecution<vt_, it_, 1, 1, 1, gpu_data_s<vt_,it_>>;
    using super::N;
    using super::host_data;
    using super::host_indices;
    using super::device_data_ptrs;
    using super::device_indices_ptrs;
    using super::gpu_named_data;

    
    IKernelExecution_Test(unsigned long long n) : super(n) {}

    void set_extra_params(int i)
    {
        local_i = i;
    }

    const auto& get_cpu_data_vector()         { return host_data; }
    const auto& get_cpu_indices_vector()      { return host_indices; }
    const auto& get_gpu_data_ptrs_vector()    { return device_data_ptrs; }
    const auto& get_gpu_indices_ptrs_vector() { return device_indices_ptrs; }
    
  private:
    void initInputsCpu() override 
    {
        for(int i=0; i< N; ++i)
        {
            host_data[0].push_back(static_cast<value_t>(i));
            host_data[1].push_back(static_cast<value_t>(0));
        }
    }

    void initIndicesCpu() override 
    {
        for(int i=0; i<N; ++i)
        {
            host_indices[0].push_back(static_cast<index_t>(i));
        }        
    }

    void setGpuNamedData() override 
    {
        gpu_named_data.input = device_data_ptrs[0];
        gpu_named_data.output = device_data_ptrs[1];
        gpu_named_data.indices = device_indices_ptrs[0];
    }

    int local_i = 0;
};

TEST(IIKernelExecutionTests, Construct) {
    using IKernelExecution_t = IKernelExecution_Test<float, int>;
    IKernelExecution_t data(4);
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();
    
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    ASSERT_EQ(cpu_indices_vector.size(), 1);
    EXPECT_EQ(cpu_indices_vector[0].size(), 0);

    ASSERT_EQ(gpu_data_ptrs_vector.size(), 2);
    EXPECT_EQ(gpu_data_ptrs_vector[0], nullptr);
    EXPECT_EQ(gpu_data_ptrs_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_ptrs_vector.size(), 1);
    EXPECT_EQ(gpu_indices_ptrs_vector[0], nullptr);

    EXPECT_EQ(data.gpu_named_data.input, nullptr);
    EXPECT_EQ(data.gpu_named_data.output, nullptr);
    EXPECT_EQ(data.gpu_named_data.indices, nullptr);
}

TEST(IIKernelExecutionTests, Initialize) {
    using IKernelExecution_t = IKernelExecution_Test<float, int>;
    size_t data_size = 4;
    IKernelExecution_t data(data_size);
    
    ASSERT_TRUE(data.init(0));

    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();

    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), data_size);
    EXPECT_EQ(cpu_data_vector[1].size(), data_size);

    ASSERT_EQ(gpu_data_ptrs_vector.size(), 2);
    EXPECT_NE(gpu_data_ptrs_vector[0], nullptr);
    EXPECT_NE(gpu_data_ptrs_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_ptrs_vector.size(), 1);
    EXPECT_NE(gpu_indices_ptrs_vector[0], nullptr);

    EXPECT_EQ(data.gpu_named_data.input, gpu_data_ptrs_vector[0]);
    EXPECT_EQ(data.gpu_named_data.output, gpu_data_ptrs_vector[1]);
    EXPECT_EQ(data.gpu_named_data.indices, gpu_indices_ptrs_vector[0]);
}

TEST(IIKernelExecutionTests, Uninitialize) {
    using IKernelExecution_t = IKernelExecution_Test<float, int>;
    size_t data_size = 4;
    IKernelExecution_t data(data_size);
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();

    ASSERT_TRUE(data.init(0));
    data.uninit();
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    ASSERT_EQ(cpu_indices_vector.size(), 1);
    EXPECT_EQ(cpu_indices_vector[0].size(), 0);

    ASSERT_EQ(gpu_data_ptrs_vector.size(), 2);
    EXPECT_EQ(gpu_data_ptrs_vector[0], nullptr);
    EXPECT_EQ(gpu_data_ptrs_vector[1], nullptr);

    ASSERT_EQ(gpu_indices_ptrs_vector.size(), 1);
    EXPECT_EQ(gpu_indices_ptrs_vector[0], nullptr);

    EXPECT_EQ(data.gpu_named_data.input, nullptr);
    EXPECT_EQ(data.gpu_named_data.output, nullptr);
    EXPECT_EQ(data.gpu_named_data.indices, nullptr);
}

*/