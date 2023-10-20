/**
 * @file SimpleKernelFileExecutionTests.cu
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


#include "execution/SimpleKernelExecution.hpp"
#include "data/SimpleKernelData.hpp"
#include "execution/JitifyCache.cuh"
#include <execution/kc_jitify.hpp>
#include "DirectoryFinder.h"

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


class SimpleKernelFileExecutionTests : public ::testing::Test // : public IKernelExecution<...>
{
public:
    // using IKernelExecution_Test_t = IKernelExecution_Test<kernel_param_types...>;
    // using super = IKernelExecution<kenrel_param_types...>
    SimpleKernelFileExecutionTests() = default;  //: super(n) {}
    ~SimpleKernelFileExecutionTests() = default; //: super(n) {}

    void SetUp()
    {

        data_ptr = std::make_shared<kc::SimpleKernelData<>>(data_size);
        data_ptr->setInitInputsFunc(whole_numbers<>);
        // data_ptr->init(0);
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

TEST_F(SimpleKernelFileExecutionTests, FromFile)
{
    auto simple_copy_file = kc::utilities::find_parent_dir_by_name("KernelColonel") / "tests" / "cpp" / "simple_copy.cuh";
    // program = kc::globalJitifyCache().program((test_dir / "cpp" / "simple_copy.cuh").str());
    program = kc::globalJitifyCache().program(simple_copy_file.string());

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

    kc::SimpleKernelExecution<> exec("simple_copy_kernel", program, check_lambda);

    dim3 grid(1);
    dim3 block(1);

    auto time = exec.time_single_execution(data_ptr, grid, block);
    ASSERT_TRUE(exec.check(data_ptr));
    std::cout << "Executing " << "simple_copy_kernel" << " with grid=" << grid << " and block="<<block << " took " << time << "ms" << std::endl;
}
