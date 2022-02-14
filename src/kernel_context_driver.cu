// driver.cpp
// simple driver file for kernel testing

// local files
#include <kernel_types.h>
#include <driver.h>
#include <device_props.h>
#include <overlap_index_access_with_data.cu>

#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define DEBUG


using vt = double;
using std::cout;
using std::endl;
using std::string;

#define N (32*32*32 * 32 * 8)

int main() {
    cout << "Processing " << N  << " elements" << endl;
    typedef ArrayCopyContext<vt, int> copy_kernel_t;
    typedef MicrobenchmarkDriver<copy_kernel_t> copy_driver_t;

    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, false>> indirection_driver_direct_t;
    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, true>> indirection_driver_indirect_t;
    
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 1>> overlapped_access_driver_1_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 2>> overlapped_access_driver_2_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 4>> overlapped_access_driver_4_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 8>> overlapped_access_driver_8_t;

    device_context dev_ctx;
    if(!dev_ctx.init()) return -1;

    std::vector<int> bs_vec;
    // Only one of the next two lines 
    // for (int bs = 32; bs <= 1024; bs *= 2) { bs_vec.push_back(bs);}
    bs_vec.push_back(128);


    string output_dir = "../../output/02-14-22_12-00/";
    
    copy_driver_t copy_driver(N, bs_vec, output_dir+"copy_kernel_output.csv", dev_ctx, true);
    copy_driver.check_then_run_kernels();
    
    indirection_driver_direct_t direct_driver(N, bs_vec,output_dir+"direct_kernel_output.csv", dev_ctx, true);
    direct_driver.check_then_run_kernels();
    indirection_driver_indirect_t indirect_driver(N, bs_vec, output_dir+"indirect_kernel_output.csv", dev_ctx, true);
    indirect_driver.check_then_run_kernels();


    overlapped_access_driver_1_t overlapped_1_driver(N, bs_vec, output_dir+"overlapped_kernel_output_1.csv", dev_ctx, true);
    overlapped_1_driver.check_then_run_kernels();

    overlapped_access_driver_2_t overlapped_2_driver(N, bs_vec, output_dir+"overlapped_kernel_output_2.csv", dev_ctx, true);
    overlapped_2_driver.check_then_run_kernels();

    overlapped_access_driver_4_t overlapped_4_driver(N, bs_vec, output_dir+"overlapped_kernel_output_4.csv", dev_ctx, true);
    overlapped_4_driver.check_then_run_kernels();

    overlapped_access_driver_8_t overlapped_8_driver(N, bs_vec, output_dir+"overlapped_kernel_output_8.csv", dev_ctx, true);
    overlapped_8_driver.check_then_run_kernels();

    return 0;
}