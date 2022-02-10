// driver.cpp
// simple driver file for kernel testing

// local files
#include <kernel_types.h>
#include <driver.h>
#include <device_props.h>

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DEBUG


using vt = double;
using std::cout;
using std::endl;
using std::string;

#define N (32*32*32)

int main() {
    cout << "Processing " << N * sizeof(vt) / 1024 * 2 << " KB of data (1/2 reads; 1/2 writes)" << endl;
    typedef ArrayCopyContext<vt, int> copy_kernel_t;
    typedef MicrobenchmarkDriver<copy_kernel_t> copy_driver_t;

    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, false>> indirection_driver_direct_t;
    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, true>> indirection_driver_indirect_t;

    device_context dev_ctx;
    if(!dev_ctx.init()) return -1;

    std::vector<int> bs_vec;
    // Only one of the next two lines 
    // for (int bs = 32; bs <= 1024; bs *= 2) { bs_vec.push_back(bs);}
    bs_vec.push_back(128);

    copy_driver_t copy_driver(N, bs_vec, "../../output/copy_kernel_output.csv", dev_ctx, true);
    copy_driver.check_then_run_kernels();
    
    indirection_driver_direct_t direct_driver(N, bs_vec, "../../output/direct_kernel_output.csv", dev_ctx, true);
    direct_driver.check_then_run_kernels();
    indirection_driver_indirect_t indirect_driver(N, bs_vec, "../../output/indirect_kernel_output.csv", dev_ctx, true);
    indirect_driver.check_then_run_kernels();

    return 0;
}