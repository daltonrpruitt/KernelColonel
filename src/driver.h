#pragma once
/**
 * @file driver.h
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Class to drive testing of kernels, including verification
 * @version 0.1
 * @date 2022-02-04
 * 
 */

#include <kernel_types.h>
#include <local_cuda_utils.h>
#include <stats.h>
#include <device_props.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>


using std::vector;
using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::ofstream;
using std::setw;


template <typename kernel_ctx_t>
class MicrobenchmarkDriver {
   private:
    ofstream output_file;
    vector<kernel_ctx_t*> contexts;

    int N = 0;
    int bs = 0;
    int kernel_runs = 50;
    int kernel_checks = 2;

   public:
    MicrobenchmarkDriver(int N, vector<int>& bs_vec, string output_filename, device_context& dev_ctx, bool span_occupancies=false) {
        dev_ctx.init();

        for (int bs : bs_vec) {
            kernel_ctx_t* curr_ctx = new kernel_ctx_t(N, bs, dev_ctx);
            contexts.push_back(curr_ctx);
            if(span_occupancies) {
                vector<int> shdmem_allocs = curr_ctx->shared_memory_allocations();
#ifdef DEBUG
                cout << "Valid ShdMem alloc amounts for "<< curr_ctx->name <<": ";
                for(int x : shdmem_allocs)
#endif
                for(int i=0; i < shdmem_allocs.size()-1; ++i){ // not last one because we already have that one (max occupancy)
                    contexts.push_back(new kernel_ctx_t(N, bs, dev_ctx, shdmem_allocs[i]));
                }
            }
        }


        // Guard against overwriting data
        struct stat output_file_buffer;
        int i = 0;
        string original_filename = output_filename;
        while (stat(output_filename.c_str(), &output_file_buffer) == 0)
        {
            cerr << "The file '" << output_filename << "' already exists in the output directory!" << endl;
            ++i;
            output_filename = original_filename+"("+to_string(i)+")";
        }

        // output_file << "Array_size,tpb,ept,bwss,twss,num_blocks,fraction_of_l2_used_per_block,num_repeat,theoretical_bandwidth"
        //              << ",shuffle_type,kernel_type,blocks_per_sm,min,med,max,avg,stddev,achieved_throughput" << endl ;
        output_file.open(output_filename.c_str());
        output_file << "kernel_type,array_size,tpb,min,med,max,avg,stddev" << endl;
    }
    ~MicrobenchmarkDriver() {
        for (auto ctx : contexts) {
            if (ctx)
                delete ctx;
        }
        output_file.close();
    }

    bool check_kernels() {
        bool pass = true;
        for (auto ctx : contexts) {
            ctx->init();
            for (int i = 0; i < kernel_checks; ++i) {
                pass = (pass && ctx->run_and_check());
            }
            ctx->uninit();
        }
        if (!pass) {
            cerr << "One or more kernels failed check!" << endl;
        }
        return pass;
    }

    void run_kernels() {
        for (auto ctx : contexts) {
#ifdef DEBUG
            cout << "Running " << ctx->name << endl;
#endif

            if(!ctx->init()) {return;}
            vector<float> times;
            for (int i = 0; i < kernel_runs; ++i) {
                float t = ctx->run();
                times.push_back(t);
            }
            ctx->uninit();
            // avg/std dev/ min, max, med
            vector<float> timing_stats = stats_from_vec(times);
#ifdef DEBUG
            cout << "Actual runtimes:" << endl;
            for (int i = 0; i < times.size(); ++i) {
                cout << setw(10) << times[i];
                if (i % 10 == 9) {
                    cout << endl;
                }
            }
            cout << endl;

            int w = 15;
            cout << "Timing stats:" << endl;
            cout << setw(w) << "min"
                      << setw(w) << "med"
                      << setw(w) << "max"
                      << setw(w) << "avg"
                      << setw(w) << "stddev" << endl;
            for (auto v : timing_stats) {
                cout << setw(w) << v;
            }
            cout << endl
                      << endl;
#endif
            // output to file
            write_data(ctx, timing_stats);
        }
    }

    void check_then_run_kernels() {
        bool pass = check_kernels();
        if (pass) {
            run_kernels();
        } else {
            cerr << "Not running kernels!" << endl;
        }
    }

    // output_file << "kernel_type,array_size,tpb,min,med,max,avg,stddev" << endl ;
    void write_data(kernel_ctx_t* ctx, vector<float> data) {
        stringstream s;
        copy(data.begin(), data.end(), std::ostream_iterator<float>(s, ","));  // https://stackoverflow.com/questions/9277906/stdvector-to-string-with-custom-delimiter
        string wo_last_comma = s.str();
        wo_last_comma.pop_back();  // https://stackoverflow.com/questions/2310939/remove-last-character-from-c-string
        output_file << ctx->name << "," << ctx->N << "," << ctx->Bsz << "," << wo_last_comma << endl;
    }
};
