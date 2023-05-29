#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "check_cuda.cuh"


#include "data/SimpleKernelData.hpp"
using namespace KernelColonel;

template<typename value_t, typename index_t>
class SimpleKernelData_SimpleCopyTest : public SimpleKernelData<value_t, index_t>
{
  public:
    using vt_ = value_t;
    using it_ = index_t;
    using super = SimpleKernelData<vt_, it_>;
    using super::N;
    using super::host_data;
    using super::host_indices;
    using super::device_data_ptrs;
    using super::device_indices_ptrs;
    using super::gpu_named_data;

    SimpleKernelData_SimpleCopyTest(unsigned long long n) : super(n) {}
    ~SimpleKernelData_SimpleCopyTest() = default;

    const auto& get_cpu_data_vector()         { return host_data; }
    const auto& get_cpu_indices_vector()      { return host_indices; }
    const auto& get_gpu_data_ptrs_vector()    { return device_data_ptrs; }
    const auto& get_gpu_indices_ptrs_vector() { return device_indices_ptrs; }
    

};

template<typename vt>
std::vector<vt> whole_numbers(unsigned int length)
{
    std::vector<vt> out;
    for(unsigned int i=0; i < length; i++)
        out.push_back(static_cast<vt>(i));
    return out;
}

template<typename vt>
bool arrays_equal(unsigned int length, std::vector<vt> inputs, std::vector<vt> outputs)
{
    for(unsigned int i=0; i < length; i++) {
        if(inputs[i] != outputs[i] && abs(inputs[i] - outputs[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

TEST(SimpleCopyTests, InitializeWithoutMethodsSet) {
    SimpleKernelData_SimpleCopyTest<float, int> data(pow(2,10));
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();
    
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    EXPECT_THROW( { data.init(0); },  std::runtime_error );
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);
}

TEST(SimpleCopyTests, InitializeWithMethodsSet) {
    SimpleKernelData_SimpleCopyTest<float, int> data(pow(2,10));
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    const auto& cpu_indices_vector = data.get_cpu_indices_vector();
    const auto& gpu_data_ptrs_vector = data.get_gpu_data_ptrs_vector();
    const auto& gpu_indices_ptrs_vector = data.get_gpu_indices_ptrs_vector();
    
    ASSERT_EQ(cpu_data_vector.size(), 2);
    EXPECT_EQ(cpu_data_vector[0].size(), 0);
    EXPECT_EQ(cpu_data_vector[1].size(), 0);

    data.setInitInputsFunc(whole_numbers<float>);

    ASSERT_TRUE(data.init(0));
    EXPECT_EQ(cpu_data_vector[0].size(), 1024);
}

template<typename gpu_data_t>
__global__
void copy_kernel_print(unsigned long long size, gpu_data_t data_struct) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size) return;
    auto input = data_struct.input;
    auto output = data_struct.output;
    if(threadIdx.x==0){
        printf("Hello \n"); // ss=%4.2f \n", 4.05);
    }
    output[idx] = input[idx];
}

TEST(SimpleCopyTests, PassToKernel) {
    using namespace ::testing;
    using KernelData_t = SimpleKernelData_SimpleCopyTest<float, int>;
    constexpr unsigned int data_size = 8;
    KernelData_t data(data_size);
    

    data.setInitInputsFunc(whole_numbers<float>);

    ASSERT_TRUE(data.init(0));
    std::vector<float> v(data_size, 0);
    // ASSERT_EQ(cpu_data_vector[1].size(), 0);
    // SimpleKernelData_gpu_data_s<float, int> dummy;
    // std::cout << "before primary error check"<< std::endl;
    // gpuErrchk( cudaPeekAtLastError() );
    // std::cout << "after primary error check/before kernel call"<< std::endl;
    copy_kernel_print<decltype(data.gpu_named_data)><<<1,8>>>(data_size, data.gpu_named_data);
    // std::cout << "after kernel call"<< std::endl;
    
    // copy_kernel<decltype(data.gpu_named_data)><<<1,4>>>(data_size, data.gpu_named_data);
    /*<SimpleKernelData_gpu_data_s<float, int>*/
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk( cudaPeekAtLastError() );
    data.copyOutputFromDeviceToHost();
    const auto& cpu_data_vector = data.get_cpu_data_vector();
    
    v.clear();
    for(int i=0; i < data_size; ++i) v.push_back(i);
    ASSERT_THAT(cpu_data_vector[1], Pointwise(FloatEq(), v));
}
