// spmv.cu
// simple driver file for spmv kernel testing

// #define DEBUG

using vt = double;
using it = uint;


// local files
#include <driver.h>
#include <device_props.h>
#include <kernels/spmv/spmv_driver.h>
#include <kernels/spmv/spmv_base.cu>
#include <kernels/spmv/spmv_la_1.cu>
#include <kernels/spmv/spmv_la_2_val4.cu>

#include <output.h>
#include <utils.h>

#include <iostream>
#include <string>
#include <algorithm>
#include <time.h>
#include <filesystem>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define XSTRINGIFY( x ) STRINGIFY ( x )
#define STRINGIFY( x ) #x

using std::cout;
using std::endl;
using std::string;
using std::to_string;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
    timespec mainStart, mainEnd;
    clock_gettime(CLOCK_MONOTONIC, &mainStart);
    int total_runs = 0;
    
    bool profile = false;
    if(argc == 2) {
        if(strcmp(argv[1],"-p") == 0 ){
            cout << "Profiling only!" << endl;    
            profile = true;
        }
    }

    device_context dev_ctx;
    if(!dev_ctx.init()) return -1;
    
    bool span_occupancies = true;

    Output output_dir;
    if(output_dir.empty()) {
        cerr << "Not continuing!" << endl;
        return -1;
    }

    fs::path base_dir = fs::path(output_dir + "");
    for(int i=0; i < 4; ++i) base_dir = base_dir.parent_path();
    fs::path matrices_dir = fs::path(base_dir).append("matrices");
    cout << "Matrices dir =" << matrices_dir << endl;

    fs::path grids_dir = fs::path(base_dir).append("grids/cell_based");
    cout << "Grids dir =" << grids_dir << endl;
    if(!fs::exists(grids_dir)) {
        cerr << "Expect directory for grid-derived matrix files names 'grids/cell_based' in main directory!"<< endl;
        exit(EXIT_FAILURE);
    }
    /*
    for (auto const& dir_entry : fs::directory_iterator{matrices_dir}) {
        string mtx_file_string = dir_entry.path().string();
        if(mtx_file_string.find(string(".mtx")) == string::npos){
            continue;
        }
        cout << "Processing " << mtx_file_string << " starting at run " << total_runs << endl;

        if(profile) {
            #include <kernels/spmv/tests/spmv_la_1_profile.test>
        } else {
            #include <kernels/spmv/tests/spmv_la_1.test>
        }

    }
    */

    for (auto const& dir_entry : fs::directory_iterator{grids_dir}) {
        string mtx_file_string = dir_entry.path().string();
        if(mtx_file_string.find(string(".crs")) == string::npos){
            continue;
        }
        cout << "Processing " << mtx_file_string << " starting at run " << total_runs << endl;
        #include <kernels/spmv/tests/spmv_la_2.test>
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
