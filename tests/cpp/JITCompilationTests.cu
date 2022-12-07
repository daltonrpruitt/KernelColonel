/**
 * @file JITCompilationTests.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Set of unit/integration tests for use of JIT compilation of kernels
 * 
 * Meant to help alleviate need for compile-time knowledge of various kernel parameters, such as those 
 * discussed in [this Medium article](https://medium.com/gpgpu/cuda-jit-compilation-1fb4950c67bb) and in 
 * [this GTC 2017 talk](https://on-demand.gputechconf.com/gtc/2017/videos/s7716-barsdell-ben-jitify.mp4).
 * Uses the repo discussed in the talk: [jitify](https://github.com/NVIDIA/jitify). 
 * The jitify repo has a BSD-3-Clause license. 
 * 
 * @version 0.1
 * @date 2022-12-03
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <jitify.hpp>


TEST(JITCompilationTest, Include) {
    ASSERT_TRUE(true);
}
