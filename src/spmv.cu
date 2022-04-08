// spmv.cu
// simple driver file for spmv kernel testing

#define DEBUG

using vt = double;
using it = uint;


// local files
#include <driver.h>
#include <device_props.h>
#include <kernels/spmv/spmv_driver.h>
#include <kernels/spmv/spmv_base.cu>
#include <kernels/spmv/spmv_la_1.cu>

#include <output.h>
#include <utils.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <time.h>

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
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
    int total_runs = 0;
    

    device_context dev_ctx;
    if(!dev_ctx.init()) return -1;
    
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


    int max_matrix_filename_id = matrix_filenames.size();

    for(int i=0; i < max_matrix_filename_id; ++i) {
        SpmvDriver basic_spmv_driver(i, 64, output_dir+"spmv.csv", &dev_ctx, false);
        basic_spmv_driver.check_then_run_kernels();
        total_runs += basic_spmv_driver.get_total_runs();
    }

    clock_gettime(CLOCK_MONOTONIC, &mainEnd);
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
