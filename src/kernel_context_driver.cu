// driver.cpp
// simple driver file for kernel testing

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

// #include <local_cuda_utils.h>

#include <kernel_test.cu>
#include <kernel_context.cu>

using vt = double;
using std::string;
using std::cout;
using std::endl;

#define N (32*32*32)

int main() {
  TestKernelContext<vt, int, N, 128> ctx;
  ctx.execute();
  bool res = ctx.check_result();
  cout << ctx.name <<  " " << (res ? "Passed" : "Failed") << "!" << endl;
 
}