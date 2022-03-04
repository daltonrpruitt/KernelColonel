// driver.cpp
// simple driver file for kernel testing

#define DEBUG
// local files
#include <driver.h>
#include <device_props.h>
#include <kernels/general/copy.cu>
#include <kernels/general/simple_indirection.cu>
#include <kernels/general/overlap_index_access_with_data.cu>
#include <kernels/general/computation.cu>
#include <kernels/burst_mode/interleaved_copy.cu>
#include <kernels/uncoalesced_cached_access/uncoalesced_reuse.cu>
#include <kernels/uncoalesced_cached_access/uncoalesced_reuse_general_size.cu>
#include <kernels/uncoalesced_cached_access/uncoalesced_reuse_general_size_single_element.cu>
#include <kernels/burst_mode/interleaved_copy_full_life.cu>

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

// #define N (32*32*32 * 32 * 8)

int main() {
    timespec mainStart, mainEnd;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
    int total_runs = 0;

    /*
    typedef ArrayCopyContext<vt, int> copy_kernel_t;
    typedef MicrobenchmarkDriver<copy_kernel_t> copy_driver_t;

    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, false>> indirection_driver_direct_t;
    typedef MicrobenchmarkDriver<SimpleIndirectionKernel<vt, int, true>> indirection_driver_indirect_t;
    
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 1>> overlapped_access_driver_1_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 2>> overlapped_access_driver_2_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 4>> overlapped_access_driver_4_t;
    typedef MicrobenchmarkDriver<OverlappedIdxDataAccessKernel<vt, int, 8>> overlapped_access_driver_8_t;
    */

    device_context dev_ctx;
    if(!dev_ctx.init()) return -1;
    unsigned long long min_array_size = dev_ctx.props_.l2CacheSize / sizeof(vt) * 40 / dev_ctx.props_.multiProcessorCount;
    min_array_size = pow(2, ceil(log2(min_array_size)));
    unsigned long long N = min_array_size * dev_ctx.props_.multiProcessorCount;

    cout << "Processing " << N  << " elements" << endl;


    std::vector<int> bs_vec;
    // Only one of the next two lines 
    // for (int bs = 256; bs <= 1024; bs *= 2) { bs_vec.push_back(bs);}
    bs_vec.push_back(64);
    bs_vec.push_back(128);
    bs_vec.push_back(1024);
    bool span_occupancies = true;

    Output output_dir;
    if(output_dir.empty()) {
        cerr << "Not continuing!" << endl;
        return -1;
    }

/*
    {
    copy_driver_t copy_driver(N, bs_vec, output_dir+"copy_kernel_output.csv", &dev_ctx, span_occupancies);
    if (!copy_driver.check_then_run_kernels()) {return -1;} 
    total_runs += copy_driver.get_total_runs();
    }
    
    {
    indirection_driver_direct_t direct_driver(N, bs_vec,output_dir+"direct_kernel_output.csv", &dev_ctx, span_occupancies);
    if (!direct_driver.check_then_run_kernels()) {return -1;} 
    indirection_driver_indirect_t indirect_driver(N, bs_vec, output_dir+"indirect_kernel_output.csv", &dev_ctx, span_occupancies);
    if (!indirect_driver.check_then_run_kernels()) {return -1;} 
    total_runs += direct_driver.get_total_runs() + indirect_driver.get_total_runs();
    }
*/

    // overlapped_access_driver_1_t overlapped_1_driver(N, bs_vec, output_dir+"overlapped_1_kernel_output.csv", &dev_ctx, true);
    // if (!overlapped_1_driver.check_then_run_kernels()) {return -1;} 
    // overlapped_access_driver_2_t overlapped_2_driver(N, bs_vec, output_dir+"overlapped_2_kernel_output.csv", &dev_ctx, true);
    // if (!overlapped_2_driver.check_then_run_kernels()) {return -1;} 
    // overlapped_access_driver_4_t overlapped_4_driver(N, bs_vec, output_dir+"overlapped_4_kernel_output.csv", &dev_ctx, true);
    // if (!overlapped_4_driver.check_then_run_kernels()) {return -1;} 
    // overlapped_access_driver_8_t overlapped_8_driver(N, bs_vec, output_dir+"overlapped_8_kernel_output.csv", &dev_ctx, true);
    // if (!overlapped_8_driver.check_then_run_kernels()) {return -1;} 
    // total_runs += overlapped_1_driver.get_total_runs() + overlapped_2_driver.get_total_runs() + 
    //                         overlapped_4_driver.get_total_runs() + overlapped_8_driver.get_total_runs();


    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 1>> comp_intens_1_driver(N, bs_vec, output_dir+"computational_intensity_1_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_1_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 2>> comp_intens_2_driver(N, bs_vec, output_dir+"computational_intensity_2_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_2_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 4>> comp_intens_4_driver(N, bs_vec, output_dir+"computational_intensity_4_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_4_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 8>> comp_intens_8_driver(N, bs_vec, output_dir+"computational_intensity_8_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_8_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 16>> comp_intens_16_driver(N, bs_vec, output_dir+"computational_intensity_16_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_16_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 32>> comp_intens_32_driver(N, bs_vec, output_dir+"computational_intensity_32_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_32_driver.check_then_run_kernels()) {return -1;} 
    // MicrobenchmarkDriver<ComputationalIntensityContext<vt, int, 64>> comp_intens_64_driver(N, bs_vec, output_dir+"computational_intensity_64_kernel_output.csv", &dev_ctx, true);
    // if (!comp_intens_64_driver.check_then_run_kernels()) {return -1;} 

    // total_runs += comp_intens_1_driver.get_total_runs() + comp_intens_2_driver.get_total_runs() + 
    //                         comp_intens_4_driver.get_total_runs() + comp_intens_8_driver.get_total_runs() + 
    //                         comp_intens_16_driver.get_total_runs() + comp_intens_32_driver.get_total_runs() + 
    //                         comp_intens_64_driver.get_total_runs();
#define XSTRINGIFY( x ) STRINGIFY ( x )
#define STRINGIFY( x ) #x

/*
#define INTER_DRIVER(X, Y) interleaved_copy_ ## X  ## _ ## Y ## _driver
#define INTERLEAVED(X, Y) { MicrobenchmarkDriver<InterleavedCopyContext<vt, int, X, Y>> \
      INTER_DRIVER(X, Y)(N, bs_vec, output_dir+ XSTRINGIFY( INTER_DRIVER(X, Y) ) ".csv", &dev_ctx, span_occupancies); \
    if (!INTER_DRIVER(X, Y).check_then_run_kernels()) {return -1;}  \
    total_runs += INTER_DRIVER(X, Y).get_total_runs(); }

   
    
    INTERLEAVED(1, 1)
    INTERLEAVED(2, 1)
    INTERLEAVED(4, 1)
    INTERLEAVED(8, 1)
    INTERLEAVED(16, 1)
    INTERLEAVED(32, 1)
    // INTERLEAVED(64, 1)

    // INTERLEAVED(1, 1)
    INTERLEAVED(1, 2)
    INTERLEAVED(1, 4)
    INTERLEAVED(1, 8)
    INTERLEAVED(1, 16)
    INTERLEAVED(1, 32)
    // INTERLEAVED(1, 64)

    // INTERLEAVED(8, 1, 1)
    INTERLEAVED(8, 2)
    INTERLEAVED(8, 4)
    INTERLEAVED(8, 8)
    INTERLEAVED(8, 16)
    INTERLEAVED(8, 32)

    // INTERLEAVED(32, 1)
    INTERLEAVED(32, 2)
    INTERLEAVED(32, 4)
    INTERLEAVED(32, 8)
    INTERLEAVED(32, 16)
*/
#define INTER_FULL_LIFE_DRIVER(X) interleaved_copy_full_life_ ## X  ## _driver
#define INTERLEAVED_FULL_LIFE(X) { MicrobenchmarkDriver<InterleavedCopyFullLifeContext<vt, int, X>> \
      INTER_FULL_LIFE_DRIVER(X)(N, bs_vec, output_dir+ XSTRINGIFY( INTER_FULL_LIFE_DRIVER(X) ) ".csv", &dev_ctx, span_occupancies); \
    if (!INTER_FULL_LIFE_DRIVER(X).check_then_run_kernels()) {return -1;}  \
    total_runs += INTER_FULL_LIFE_DRIVER(X).get_total_runs(); }
    
    unsigned long long tmp_n = N;

    INTERLEAVED_FULL_LIFE(1)
    INTERLEAVED_FULL_LIFE(2)
    INTERLEAVED_FULL_LIFE(4)
    INTERLEAVED_FULL_LIFE(8)
    INTERLEAVED_FULL_LIFE(16)
    INTERLEAVED_FULL_LIFE(32)

/*
    N = tmp_n / 8 * 9;
    INTERLEAVED_FULL_LIFE(6)
    INTERLEAVED_FULL_LIFE(12)
    INTERLEAVED_FULL_LIFE(24)
    INTERLEAVED_FULL_LIFE(18)

    N = tmp_n / 4 * 5;
    INTERLEAVED_FULL_LIFE(10)
    INTERLEAVED_FULL_LIFE(20)
    N = N / 4 * 3;
    INTERLEAVED_FULL_LIFE(30)

    N = tmp_n / 8 * 7;
    INTERLEAVED_FULL_LIFE(14)
    INTERLEAVED_FULL_LIFE(28)

    N = tmp_n / 8 * 11;
    INTERLEAVED_FULL_LIFE(22)

    N = tmp_n / 16 * 13;
    INTERLEAVED_FULL_LIFE(26)
    N = tmp_n;
*/

//*/
/*
#define UNCOAL_REUSE_DRIVER(B1, B2) uncoalesced_reuse_ ## B1  ## _ ## B2 ## _driver

#define UNCOAL_REUSE(B1, B2) { MicrobenchmarkDriver<UncoalescedReuseContext<vt, int, B1, B2>> \
      UNCOAL_REUSE_DRIVER(B1, B2)(N, bs_vec, output_dir+ XSTRINGIFY( UNCOAL_REUSE_DRIVER(B1, B2) ) ".csv", &dev_ctx, span_occupancies); \
    if (!UNCOAL_REUSE_DRIVER(B1, B2).check_then_run_kernels()) {return -1;}  \
    total_runs += UNCOAL_REUSE_DRIVER(B1, B2).get_total_runs(); }
    
    UNCOAL_REUSE(false, false)
    UNCOAL_REUSE(true, false)
    UNCOAL_REUSE(false, true)
    UNCOAL_REUSE(true, true)
//*/


/*

#define UNCOAL_REUSE_GENERAL_DRIVER(B1, B2, X) uncoalesced_reuse_ ## B1  ## _ ## B2 ## _ ## X ## _driver

#define UNCOAL_REUSE_GENERAL(B1, B2, X) { MicrobenchmarkDriver<UncoalescedReuseGeneralContext<vt, int, B1, B2, X>> \
      UNCOAL_REUSE_GENERAL_DRIVER(B1, B2, X)(N, bs_vec, output_dir+ XSTRINGIFY( UNCOAL_REUSE_GENERAL_DRIVER(B1, B2, X) ) ".csv", &dev_ctx, span_occupancies); \
    if (!UNCOAL_REUSE_GENERAL_DRIVER(B1, B2, X).check_then_run_kernels()) {return -1;}  \
    total_runs += UNCOAL_REUSE_GENERAL_DRIVER(B1, B2, X).get_total_runs(); }
    
    UNCOAL_REUSE_GENERAL(false, false, 1024)
    UNCOAL_REUSE_GENERAL(true, false, 1024)
    UNCOAL_REUSE_GENERAL(false, true, 1024)
    UNCOAL_REUSE_GENERAL(true, true, 1024)
    
    UNCOAL_REUSE_GENERAL(false, false, 2048)
    UNCOAL_REUSE_GENERAL(true, false, 2048)
    UNCOAL_REUSE_GENERAL(false, true, 2048)
    UNCOAL_REUSE_GENERAL(true, true, 2048)

    UNCOAL_REUSE_GENERAL(false, false, 4096)
    UNCOAL_REUSE_GENERAL(true, false, 4096)
    UNCOAL_REUSE_GENERAL(false, true, 4096)
    UNCOAL_REUSE_GENERAL(true, true, 4096)

    UNCOAL_REUSE_GENERAL(false, false, 8192)
    UNCOAL_REUSE_GENERAL(true, false, 8192)
    UNCOAL_REUSE_GENERAL(false, true, 8192)
    UNCOAL_REUSE_GENERAL(true, true, 8192)

    UNCOAL_REUSE_GENERAL(false, false, 16384)
    UNCOAL_REUSE_GENERAL(true, false, 16384)
    UNCOAL_REUSE_GENERAL(false, true, 16384)
    UNCOAL_REUSE_GENERAL(true, true, 16384)

    UNCOAL_REUSE_GENERAL(false, false, 32768)
    UNCOAL_REUSE_GENERAL(true, false, 32768)
    UNCOAL_REUSE_GENERAL(false, true, 32768)
    UNCOAL_REUSE_GENERAL(true, true, 32768)

*/
#define UNCOAL_REUSE_GENERAL_SINGLE_DRIVER(B1, B2, X) uncoalesced_reuse_general_single_ ## B1  ## _ ## B2 ## _ ## X ## _driver

#define UNCOAL_REUSE_GENERAL_SINGLE(B1, B2, X) { MicrobenchmarkDriver<UncoalescedReuseGeneralSingleElementContext<vt, int, B1, B2, X>> \
      UNCOAL_REUSE_GENERAL_SINGLE_DRIVER(B1, B2, X)(N, bs_vec, output_dir+ XSTRINGIFY( UNCOAL_REUSE_GENERAL_SINGLE_DRIVER(B1, B2, X) ) ".csv", &dev_ctx, span_occupancies); \
    if (!UNCOAL_REUSE_GENERAL_SINGLE_DRIVER(B1, B2, X).check_then_run_kernels()) {return -1;}  \
    total_runs += UNCOAL_REUSE_GENERAL_SINGLE_DRIVER(B1, B2, X).get_total_runs(); }
    
    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 1024)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 1024)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 1024)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 1024)
    
    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 2048)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 2048)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 2048)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 2048)

    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 4096)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 4096)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 4096)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 4096)

    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 8192)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 8192)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 8192)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 8192)

    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 16384)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 16384)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 16384)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 16384)

    UNCOAL_REUSE_GENERAL_SINGLE(false, false, 32768)
    UNCOAL_REUSE_GENERAL_SINGLE(true, false, 32768)
    UNCOAL_REUSE_GENERAL_SINGLE(false, true, 32768)
    UNCOAL_REUSE_GENERAL_SINGLE(true, true, 32768)

//*/

    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
    double main_time = elapsed_time_ms(mainStart, mainEnd);
    
    cout << "#########  Finished  #########" << endl << endl;
    cout << "Total runs performed        = " << total_runs << endl;
    cout << "Total time taken (h:mm:ss)     = " << std::setfill('0') << std::setw(2) 
                                                << (int)main_time / 1000 / 60 / 60 << ":" 
                                                << std::setfill('0') << std::setw(2)
                                                << (int)main_time / 1000 / 60 << ":" 
                                                << std::setfill('0') << std::setw(2)
                                                << (int)main_time / 1000 % 60 << endl;

    return 0;
}
