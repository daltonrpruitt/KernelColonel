// driver.cpp
// simple driver file for kernel testing

#define DEBUG
// local files
#include <driver.h>
#include <device_props.h>
#include <copy.cu>
#include <simple_indirection.cu>
#include <overlap_index_access_with_data.cu>
#include <computation.cu>
#include <output.h>
#include <utils.h>

#include <iostream>
#include <string>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime_api.h>



using vt = double;
using std::cout;
using std::endl;
using std::string;
using std::to_string;

#define N (32*32*32 * 32 * 8)

int main() {
    timespec mainStart, mainEnd;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
    int total_runs = 0;

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

    Output output_dir;
    if(output_dir.empty()) {
        cerr << "Not continuing!" << endl;
        return -1;
    }
    string filename = "new_file.csv";
    cout << output_dir+"indirect_kernel_output.csv" << endl;
    cout << output_dir+filename << endl;

    copy_driver_t copy_driver(N, bs_vec, output_dir+"copy_kernel_output.csv", &dev_ctx, true);
    if (!copy_driver.check_then_run_kernels()) {return -1;} 
    total_runs += copy_driver.get_total_runs();

    
    indirection_driver_direct_t direct_driver(N, bs_vec,output_dir+"direct_kernel_output.csv", &dev_ctx, true);
    if (!direct_driver.check_then_run_kernels()) {return -1;} 
    indirection_driver_indirect_t indirect_driver(N, bs_vec, output_dir+"indirect_kernel_output.csv", &dev_ctx, true);
    if (!indirect_driver.check_then_run_kernels()) {return -1;} 
    total_runs += direct_driver.get_total_runs() + indirect_driver.get_total_runs();


    overlapped_access_driver_1_t overlapped_1_driver(N, bs_vec, output_dir+"overlapped_1_kernel_output.csv", &dev_ctx, true);
    if (!overlapped_1_driver.check_then_run_kernels()) {return -1;} 
    overlapped_access_driver_2_t overlapped_2_driver(N, bs_vec, output_dir+"overlapped_2_kernel_output.csv", &dev_ctx, true);
    if (!overlapped_2_driver.check_then_run_kernels()) {return -1;} 
    overlapped_access_driver_4_t overlapped_4_driver(N, bs_vec, output_dir+"overlapped_4_kernel_output.csv", &dev_ctx, true);
    if (!overlapped_4_driver.check_then_run_kernels()) {return -1;} 
    overlapped_access_driver_8_t overlapped_8_driver(N, bs_vec, output_dir+"overlapped_8_kernel_output.csv", &dev_ctx, true);
    if (!overlapped_8_driver.check_then_run_kernels()) {return -1;} 
    total_runs += overlapped_1_driver.get_total_runs() + overlapped_2_driver.get_total_runs() + 
                            overlapped_4_driver.get_total_runs() + overlapped_8_driver.get_total_runs();


    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 1>> comp_intens_1_driver(N, bs_vec, output_dir+"computational_intensity_1_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_1_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 2>> comp_intens_2_driver(N, bs_vec, output_dir+"computational_intensity_2_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_2_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 4>> comp_intens_4_driver(N, bs_vec, output_dir+"computational_intensity_4_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_4_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 8>> comp_intens_8_driver(N, bs_vec, output_dir+"computational_intensity_8_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_8_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 16>> comp_intens_16_driver(N, bs_vec, output_dir+"computational_intensity_16_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_16_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 32>> comp_intens_32_driver(N, bs_vec, output_dir+"computational_intensity_32_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_32_driver.check_then_run_kernels()) {return -1;} 
    MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 64>> comp_intens_64_driver(N, bs_vec, output_dir+"computational_intensity_64_kernel_output.csv", &dev_ctx, true);
    if (!comp_intens_64_driver.check_then_run_kernels()) {return -1;} 

    total_runs += comp_intens_1_driver.get_total_runs() + comp_intens_2_driver.get_total_runs() + 
                            comp_intens_4_driver.get_total_runs() + comp_intens_8_driver.get_total_runs() + 
                            comp_intens_16_driver.get_total_runs() + comp_intens_32_driver.get_total_runs() + 
                            comp_intens_64_driver.get_total_runs();


    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    double main_time = elapsed_time_ms(mainStart, mainEnd);
    
    cout << "#########  Finished  #########" << endl << endl;
    cout << "Total runs performed        = " << total_runs << endl;
    cout << "Total time taken (m:ss)     = " <<(int)main_time / 1000 / 60 << ":" << (int)main_time / 1000 % 60 << endl;

    return 0;
}