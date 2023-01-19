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
 * A lot of this is taken straight from the example in jitify (see 
 * [here](https://github.com/NVIDIA/jitify/blob/master/jitify_example.cpp)).
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
#include <tuple>

#include <cuda.h>
#include <cuda_runtime_api.h>

#ifdef LINUX  // Only supported by gcc on Linux (defined in Makefile)
#define JITIFY_ENABLE_EMBEDDED_FILES 1
#endif

#define VERBOSE
#ifdef VERBOSE
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#endif // VERBOSE
#include "jitify.hpp"

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      FAIL() << "Experienced above CUDA error!";                          \
    }                                                                     \
  } while (0)


template <typename T>
bool are_close(T in, T out) {
  return fabs(in - out) <= 1e-5f * fabs(in);
}


TEST(JITCompilationTest, SimpleProgram) {
    const char* program_source =
        "my_program\n"
        "template<int N, typename T>\n"
        "__global__\n"
        "void my_kernel(T* data) {\n"
        "    T data0 = data[0];\n"
        "    for( int i=0; i<N-1; ++i ) {\n"
        "        data[0] *= data0;\n"
        "    }\n"
        "}\n";
    static jitify::JitCache kernel_cache;
    jitify::Program program = kernel_cache.program(program_source, 0);
    
    using T = float;

    T h_data = 5;
    T* d_data;
    cudaMalloc((void**)&d_data, sizeof(T));
    cudaMemcpy(d_data, &h_data, sizeof(T), cudaMemcpyHostToDevice);
    dim3 grid(1);
    dim3 block(1);
    using jitify::reflection::type_of;
    CHECK_CUDA(program.kernel("my_kernel")
                    .instantiate(3, type_of(*d_data))
                    .configure(grid, block)
                    .launch(d_data));
    cudaMemcpy(&h_data, d_data, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    std::cout << h_data << std::endl;
    ASSERT_TRUE(are_close(h_data, 125.f));
}

/**
 * @brief What I need to test for here.
 * 
 * - can pass template parameter pack through to the kernel instantiation
 * 
 */

template<int N, typename... Ts> using NthTypeOf =
        typename std::tuple_element<N, std::tuple<Ts...>>::type;
