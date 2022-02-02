// driver.cpp
// simple driver file for kernel testing

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

// #include <local_cuda_utils.h>

#include <kernel_test.cu>
#include <kernel_types.h>

using vt = double;
using std::string;
using std::cout;
using std::endl;

#define N (32*32*32)

int main() {
  cout << "Processing " << N * sizeof(vt) / 1024 * 2 << " KB of data (1/2 reads; 1/2 writes)" << endl;
  // typedef ArrayCopyContext<vt, int> kernel_t;

// #pragma unroll
  for(int bs = 128; bs <= 1024; bs*=2) {
    ArrayCopyContext<vt, int> ctx(N, bs);
    bool res = ctx.run_and_check();
    
    cout << ctx.name <<  " bs=" << bs << " " << (res ? "Passed" : "Failed") << "!" << endl;
  }
  return 0;
 
}