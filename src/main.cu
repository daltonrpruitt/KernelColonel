// driver.cpp
// simple driver file for kernel testing
/**
 * @file main.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Showcase of how I used KernelColonel to test kernels
 * @version 0.1
 * @date 2022-07-22
 * 
 * The current design is ... less than optimal. 
 * I plan to have a more modern-ish design to allow for better and generally more sensible setup. 
 * 
 */

#define DEBUG

using vt = double;
using it = unsigned long long;
#include <indices_generation.h>


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
#include <kernels/uncoalesced_cached_access/uncoalesced_reuse_gen_single_ILP.cu>
#include <kernels/burst_mode/interleaved_copy_full_life.cu>
#include <kernels/burst_mode/interleaved_fl_ilp.cu>
#include <kernels/indirect/indirect_copy.cu>
#include <kernels/expansion_contraction/expansion_contraction.cu>

#include <output.h>
#include <utils.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <time.h>

#if defined(_WIN64)
#include <clock_gettime_windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

#define XSTRINGIFY( x ) STRINGIFY ( x )
#define STRINGIFY( x ) #x


using std::cout;
using std::endl;
using std::string;
using std::to_string;

// #define N (32*32*32 * 32 * 8)

int main(int argc, char** argv) {
    timespec mainStart, mainEnd;
    clock_gettime(0, &mainStart);
    int total_runs = 0;

    std::vector<int> inputs;
    for(int i=1; i<argc; ++i){
        inputs.push_back(atoi(argv[i]));
    } 


    GpuDeviceContext dev_ctx;
    if(!dev_ctx.init()) return -1;
    unsigned long long min_array_size = dev_ctx.props_.l2CacheSize / sizeof(vt) * 40 / dev_ctx.props_.multiProcessorCount;
    min_array_size = pow(2, ceil(log2(min_array_size)));
    unsigned long long N = min_array_size * dev_ctx.props_.multiProcessorCount;

#ifdef DEBUG
    int max_shd_mem_per_block = dev_ctx.props_.sharedMemPerBlock;
    int max_shd_mem_per_proc = dev_ctx.props_.sharedMemPerMultiprocessor;
    if(dev_ctx.props_.major == 8) {
        max_shd_mem_per_block = 164 * 1024;
        max_shd_mem_per_proc =  164 * 1024;
    }
    cout << "Shared memory data:" 
            << " mx shdmem/blk=" << max_shd_mem_per_block 
            << " mx shdmem/SM=" << max_shd_mem_per_proc 
            << endl;
#endif

    cout << "Processing " << N  << " elements" << endl;


    std::vector<int> bs_vec;
    // Only one of the next two lines 
    // for (int bs = 256; bs <= 1024; bs *= 2) { bs_vec.push_back(bs);}
    int min_block_size = dev_ctx.props_.maxThreadsPerMultiProcessor / dev_ctx.props_.maxBlocksPerMultiProcessor ;
    bs_vec.push_back(min_block_size);
    // bs_vec.push_back(128);
    // bs_vec.push_back(1024);
    bool span_occupancies = true;
    if(span_occupancies && !span_occupancies) return -1;

    Output output_dir;
    if(output_dir.empty()) {
        cerr << "Not continuing!" << endl;
        return -1;
    }

/*
    #include <kernels/general/tests/copy.test>
    #include <kernels/general/tests/indirection_simple.test>
    #include <kernels/general/tests/overlapped_access.test>
    #include <kernels/general/tests/computational_intensity.test>
    #include <kernels/burst_mode/tests/interleaved_simple.test>
    #include <kernels/burst_mode/tests/interleaved_full_life.test>
    #include <kernels/uncoalesced_cache_access/tests/uncoalesced_reuse.test>
    #include <kernels/uncoalesced_cache_access/tests/uncoalesced_reuse_general.test>
    #include <kernels/uncoalesced_cache_access/tests/uncoalesced_reuse_general_single.test>
  */  

    #include <kernels/general/tests/copy.test>

 
    // Phase 1

    //      Burst mode
    //  #include <kernels/burst_mode/tests/interleaved_full_life_ILP.test>
    // Note: keeping ILP = 1 in this one!



    //      Uncoalesced reuse
    // #include <kernels/uncoalesced_cached_access/tests/uncoalesced_reuse_general_single_ILP.test>
    

    // Phase 2

    //      Indirect Coalescing
    // #include <kernels/general/tests/indirection_simple.test>

    //      Indirect Burst Mode
    //          N/A


    // Phase 3

    //      Sector-based Uncoalesced Reuse (vs warpsize-based) - indirection only!
    // #include <kernels/indirect/tests/indirect_copy_warpsize_based_uncoalescing.test>
    // #include <kernels/indirect/tests/indirect_copy_sector_based_uncoalescing.test>
    //  #include <kernels/indirect/tests/indirect_copy_uncoalesced_no_reuse.test>

    //      Expansion-Contraction
    // #include <kernels/expansion_contraction/tests/expansion_contraction.test>

    //      SpMV Latency Amortization using sector-based uncoalesced preloading 
    //  Separate executable

    //      Bonus : Cost of direct index computation latencies (out of spmv)



    // Profiling
    // #include <kernels/indirect/tests/indirect_copy_profiling.test>
    

    clock_gettime(0, &mainEnd);
    double main_time = elapsed_time_ms(mainStart, mainEnd);
    
    cout << "#########  Finished  #########" << endl << endl;
    cout << "Total runs performed        = " << total_runs << endl;
    cout << "Total time taken (h:mm:ss)     = " << std::setfill('0') << std::setw(2) 
                                                << (int)main_time / 1000 / 60 / 60 << ":" 
                                                << std::setfill('0') << std::setw(2)
                                                << ((int)main_time / 1000 / 60 ) % 60 << ":" 
                                                << std::setfill('0') << std::setw(2)
                                                << ((int)main_time / 1000 ) % 60 << endl;

    return 0;
}
