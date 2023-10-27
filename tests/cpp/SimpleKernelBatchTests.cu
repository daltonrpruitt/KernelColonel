/**
 * @file SimpleKernelRunSetTests.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Set of unit/integration tests for the IKernelExecution using a kernel from a file
 *
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <string>
#include <memory>

// #include <cuda.h>
// #include <cuda_runtime_api.h>


#include "data/SimpleKernelData.hpp"
#include "execution/SimpleKernelExecution.hpp"
#include "execution/SimpleKernelRunSet.hpp"
#include "execution/SimpleKernelBatch.hpp"
#include "execution/JitifyCache.cuh"
#include <execution/kc_jitify.hpp>

#include "utils/utils.hpp"

namespace kc = KernelColonel;

template<typename vt = double>
std::vector<vt> whole_numbers(unsigned int length)
{
    std::vector<vt> out;
    for(unsigned int i=0; i < length; i++)
        out.push_back(static_cast<vt>(i));
    return out;
}


static std::string copy_kernel_program_name  { "simple_copy_kernel" };
static const std::string simple_copy_kernel_source_string =
    copy_kernel_program_name + 
    std::string(R"(
    template<typename value_t, typename index_t> 
    struct SimpleKernelData_gpu_data_s
    {
        value_t* input = nullptr;
        value_t* output = nullptr;
        index_t* indices = nullptr;
    };

    template<typename vt, typename it>
    __global__
    void simple_copy_kernel(unsigned int N, SimpleKernelData_gpu_data_s<vt,it> gpu_data) {
        // unsigned int x = blockIdx.x * blockSize.x + threadIdx.x;
        if(blockIdx.x != 0 || threadIdx.x != 0) return;
        for( int i=0; i<N; ++i ) {
            gpu_data.output[i] = gpu_data.input[i];
            // if(i<10) printf("At i=%d input[i]=%f, output[i]=%f\n",i, gpu_data.input[i], gpu_data.output[i]);
        }
    })");

class SimpleKernelBatchTests : public ::testing::Test // : public IKernelExecution<...>
{
public:
    SimpleKernelBatchTests() = default;
    ~SimpleKernelBatchTests() = default;

    void SetUp()
    {

        data_ptr = std::make_shared<kc::SimpleKernelData<>>(data_size);
        data_ptr->setInitInputsFunc(whole_numbers<>);
        program = kc::globalJitifyCache().program(simple_copy_kernel_source_string);
    }

    void TearDown()
    {
        data_ptr.reset();
    }

protected:
    unsigned long long data_size = 5;
    std::shared_ptr<kc::SimpleKernelData<>> data_ptr;
    jitify::Program program; 
};

TEST_F(SimpleKernelBatchTests, CreateWithData)
{
    kc::SimpleKernelBatch<> batch(data_ptr);
}



TEST_F(SimpleKernelBatchTests, AddRunSets)
{
    auto check_lambda = [&](const auto &input, const auto &output, const auto &indices)
    {
        if (input.size() != output.size())
            return false;
        for (int i = 0; i < input.size(); i++)
            if (input[i] != output[i])
            {
                std::cout << "data differs at i="<<i<<"with in[i]="<<input[i] << " and out[i]=" <<output[i] << std::endl;
                return false;
            }

        return true;
    };


    kc::SimpleKernelBatch<> batch(data_ptr);
    auto exec_ptr = std::make_shared<kc::SimpleKernelExecution<>>("simple_copy_kernel", program, check_lambda);
    
    std::vector<std::shared_ptr<kc::SimpleKernelRunSet<>>> run_set_ptrs;

    for(int i=1; i<=8; i*=2)
    {
        for(int j=1; j<=8; j*=2)
        {
            auto run_set_ptr = std::make_shared<kc::SimpleKernelRunSet<>>(exec_ptr, data_ptr, dim3(i), dim3(j));
            ASSERT_TRUE(batch.add_run_set(run_set_ptr));
        }
    }
}
/*
    kc::SimpleKernelRunSet<> run_set(exec_ptr, data_ptr, grid, block, times_run);


    auto check_lambda = [&](const auto &input, const auto &output, const auto &indices)
    {
        if (input.size() != output.size())
            return false;
        for (int i = 0; i < input.size(); i++)
            if (input[i] != output[i])
            {
                std::cout << "data differs at i="<<i<<"with in[i]="<<input[i] << " and out[i]=" <<output[i] << std::endl;
                return false;
            }

        return true;
    };


    dim3 grid(1);
    dim3 block(1);

    unsigned int times_run = 5;

    ASSERT_TRUE(run_set.check_and_run_all());
    auto times = run_set.get_run_timings();
    ASSERT_EQ(times.size(), times_run);

    std::cout << "Executing " << "simple_copy_kernel" << " with grid=" << grid << " and block=" << block << " for " << times_run << " times took the following ms timings: " << times << std::endl;
}
*/