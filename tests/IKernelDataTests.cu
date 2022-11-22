#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "IKernelData.cuh"


// Add some tests for IKernelData class !!!
// Can derive from IKernelData and perform some operations in lambdas???

template<typename value_t, typename index_t> 
class KernelData_Test : public IKernelData<value_t, index_t, 1, 1, 1>
{
  public:
    using vt_ = value_t;
    using it_ = index_t;
    using super = IKernelData<vt_, it_, 1, 1, 1>;
    using super::N;
    using super::host_data;
    using super::host_indices;
    using super::device_data_ptrs;
    using super::device_indices_ptrs;

    
    KernelData_Test(unsigned long long n) : super(n) {}
    ~KernelData_Test() = default;

    void set_extra_params(int i)
    {
        local_i = i;
    }

    const auto& get_cpu_data_vector()         { return host_data; }
    const auto& get_cpu_indices_vector()      { return host_indices; }
    const auto& get_gpu_data_ptrs_vector()    { return device_data_ptrs; }
    const auto& get_gpu_indices_ptrs_vector() { return device_indices_ptrs; }
    
    struct gpu_data_s
    {
        value_t* input = nullptr;
        value_t* output = nullptr;
        index_t* indices = nullptr;
    } gpu_named_data;

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

TEST(IKernelDataTests, Construct) {
    using KernelData_t = KernelData_Test<float, int>;
    KernelData_t data(4);
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

TEST(IKernelDataTests, Initialize) {
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t data(data_size);
    
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

TEST(IKernelDataTests, Uninitialize) {
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t data(data_size);
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

TEST(IKernelDataTests, ReinitializeWithSameDevice) {
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t data(data_size);
    
    ASSERT_TRUE(data.init(0));
    ASSERT_TRUE(data.init(0));
    data.uninit();
    ASSERT_TRUE(data.init(0));
}

TEST(IKernelDataTests, ReinitializeWithDifferentDevice) {
    int count;
    cudaGetDevice(&count);
    if(count == 1) {
        GTEST_SKIP();
    }
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t data(data_size);
    
    ASSERT_TRUE(data.init(0));
    ASSERT_TRUE(data.init(0));
    ASSERT_FALSE(data.init(1));
    data.uninit();
    ASSERT_TRUE(data.init(0));
}

TEST(IKernelDataTests, Destruct) {
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t* data_ptr = new KernelData_t(data_size);
    
    ASSERT_TRUE(data_ptr->init(0));
    ASSERT_NO_THROW( { delete data_ptr; } );
}

template<typename gpu_data_t>
__global__
void copy_kernel(unsigned long long size, gpu_data_t data_struct) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    auto input = data_struct.input;
    auto output = data_struct.output;
    output[idx] = input[idx];
}

TEST(IKernelDataTests, PassToKernel) {
    using namespace ::testing;
    using KernelData_t = KernelData_Test<float, int>;
    size_t data_size = 4;
    KernelData_t data(data_size);
    
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    ASSERT_TRUE(data.init(0));
    std::vector<float> v(data_size, 0);
    ASSERT_THAT(cpu_data_vector[1], Pointwise(FloatEq(), v));

    copy_kernel<decltype(data.gpu_named_data)><<<1,4>>>(data_size, data.gpu_named_data);
    data.copyOutputToDevice();
    
    ASSERT_THAT(cpu_data_vector[1], ElementsAre(0, 1, 2, 3));
}
