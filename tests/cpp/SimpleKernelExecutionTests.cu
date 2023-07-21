/**
 * @file SimpleKernelExecutionTests.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Set of unit/integration tests for the IKernelExecution class
 * @version 0.1
 * @date 2022-11-25
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <string>
#include <memory>

// #include <cuda.h>
// #include <cuda_runtime_api.h>


#include "execution/SimpleKernelExecution.hpp"
#include "data/SimpleKernelData.hpp"
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

std::string copy_kernel_program_name  { "simple_copy_kernel" };
const std::string simple_copy_kernel_source_string =
    std::string("simple_copy_kernel") + 
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
        for( int i=0; i<N; ++i ) {
            gpu_data.output[i] = gpu_data.input[i];
            if(i<10) printf("At i=%d input[i]=%f, output[i]=%f\n",i, gpu_data.input[i], gpu_data.output[i]);
        }
    })");

class SimpleKernelExecutionTests : public ::testing::Test // : public IKernelExecution<...>
{
public:
    // using IKernelExecution_Test_t = IKernelExecution_Test<kernel_param_types...>;
    // using super = IKernelExecution<kenrel_param_types...>
    SimpleKernelExecutionTests() = default;  //: super(n) {}
    ~SimpleKernelExecutionTests() = default; //: super(n) {}

    void SetUp()
    {

        data_ptr = std::make_shared<kc::SimpleKernelData<>>(data_size);
        data_ptr->setInitInputsFunc(whole_numbers<>);
        // data_ptr->init(0);
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

TEST_F(SimpleKernelExecutionTests, Construct)
{
    auto check_lambda = [&](const auto &input, const auto &output, const auto &indices)
    {
        return true;
    };
    EXPECT_NO_THROW( { kc::SimpleKernelExecution<> exec(copy_kernel_program_name, program, check_lambda); } );
}

TEST_F(SimpleKernelExecutionTests, ExecuteOnData)
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
    kc::SimpleKernelExecution<> exec(copy_kernel_program_name, program, check_lambda);

    dim3 grid(1);
    dim3 block(1);

    auto time = exec.time_single_execution(data_ptr, grid, block);
    ASSERT_TRUE(exec.check(data_ptr));
    std::cout << "Executing " << copy_kernel_program_name << " with grid=" << grid << " and block="<<block << " took " << time << "ms" << std::endl;
}

/**
 * @brief Helper function to repeatedly test SimpleKernelExecution::time_single_execution()
 */
template<typename value_type>
void executeCopyKernelOnNewData(unsigned long long data_size, dim3 grid, dim3 block, std::string type_string, std::string copy_kernel_program_name, jitify::Program& program)
{
    auto check_lambda = [&](const auto& input, const auto& output, const auto& indices)
    {
        if (input.size() != output.size())
            return false;
        for (int i = 0; i < input.size(); i++)
            if (input[i] != output[i])
            {
                return false;
            }
        return true;
    };

    auto new_data = std::make_shared<kc::SimpleKernelData<value_type>>(data_size);
    new_data->setInitInputsFunc(whole_numbers<value_type>);
    kc::SimpleKernelExecution<value_type> new_exec(copy_kernel_program_name, program, check_lambda);

    auto time = new_exec.time_single_execution(new_data, grid, block);
    ASSERT_TRUE(new_exec.check(new_data));
    std::cout << "Executing " << copy_kernel_program_name << " vt="<<type_string<< " with grid = " << grid << " and block = " << block << " took " << time << "ms" << std::endl;
}

TEST_F(SimpleKernelExecutionTests, ExecutionOnDifferentDataTypes)
{

    using jitify::reflection::reflect_template;

    dim3 grid(1);
    dim3 block(1);

    executeCopyKernelOnNewData<int>(data_size, grid, block, "int", copy_kernel_program_name, program);
    executeCopyKernelOnNewData<float>(data_size, grid, block, "float", copy_kernel_program_name, program);
    executeCopyKernelOnNewData<unsigned long>(data_size, grid, block, "unsigned long", copy_kernel_program_name, program);
}

