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

#include "utils/type_name.hpp"

#include "execution/SimpleKernelExecution.hpp"
#include "data/SimpleKernelData.hpp"

namespace kc = KernelColonel;

class SimpleKernelExecutionTests : public ::testing::Test // : public IKernelExecution<...>
{
public:
    // using IKernelExecution_Test_t = IKernelExecution_Test<kernel_param_types...>;
    // using super = IKernelExecution<kenrel_param_types...>
    SimpleKernelExecutionTests() = default;  //: super(n) {}
    ~SimpleKernelExecutionTests() = default; //: super(n) {}

    void SetUp()
    {
        data_ptr = std::make_shared<kc::SimpleKernelData<>>(1000);
    }

    void TearDown()
    {
    }

protected:
    std::shared_ptr<kc::SimpleKernelData<>> data_ptr;
};

TEST_F(SimpleKernelExecutionTests, Construct)
{
    auto check_lambda = [&](const auto &input, const auto &output, const auto &indices)
    {
        return true;
    };
    std::string test_name{"Test Execution"};
    kc::SimpleKernelExecution<> exec(test_name, check_lambda);
}

TEST_F(SimpleKernelExecutionTests, ExecuteOnData)
{
    auto check_lambda = [&](const auto &input, const auto &output, const auto &indices)
    {
        if (input.size() != output.size())
            return false;
        for (int i = 0; i < input.size(); i++)
            if (input[i] != output[i])
                return false;

        return true;
    };
    std::string test_name{"Test Execution"};
    kc::SimpleKernelExecution<> exec(test_name, check_lambda);

    ASSERT_TRUE(exec.check(data_ptr));
}

