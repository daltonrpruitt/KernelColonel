// driver.cpp
// simple driver file for kernel testing

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>

// #include <local_cuda_utils.h>

#define DEBUG

// local files
#include <kernel_types.h>
#include <driver.h>


using vt = double;
using std::cout;
using std::endl;
using std::string;

#define N (32*32*32)

int main() {
    cout << "Processing " << N * sizeof(vt) / 1024 * 2 << " KB of data (1/2 reads; 1/2 writes)" << endl;
    typedef ArrayCopyContext<vt, int> kernel_t;
    typedef MicrobenchmarkDriver<kernel_t> driver_t;
    device_context dev_ctx;

    std::vector<int> bs_vec; for (int bs = 32; bs <= 1024; bs *= 2) { bs_vec.push_back(bs);}
    driver_t driver(N, bs_vec, "../../output/kernel_output.csv", dev_ctx);
    driver.check_then_run_kernels();

    return 0;
}