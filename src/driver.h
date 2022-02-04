// driver.cpp
// simple driver file for kernel testing

#include <kernel_types.h>
#include <local_cuda_utils.h>
#include <stats.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>
#include <iomanip>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::string;

template <typename kernel_ctx_t>
class MicrobenchmarkDriver {
   private:
    std::ofstream output_file;
    std::vector<kernel_ctx_t *> contexts;
    int N=0;
    int bs=0;
    int kernel_runs = 50;
    int kernel_checks = 2;

   public:
    MicrobenchmarkDriver(int N, std::vector<int> bs_vec, std::string output_filename) {
        for (int bs : bs_vec) {
            contexts.push_back(new kernel_ctx_t(N, bs));
        }
        // output_file(outfilename);
        // // output_file << "Array_size,tpb,ept,bwss,twss,num_blocks,fraction_of_l2_used_per_block,num_repeat,theoretical_bandwidth" 
        // //              << ",shuffle_type,kernel_type,blocks_per_sm,min,med,max,avg,stddev,achieved_throughput" << endl ;
        output_file.open(output_filename.c_str());
        output_file << "kernel_type,array_size,tpb,min,med,max,avg,stddev" << endl ;
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
            std::cerr << "One or more kernels failed check!" << std::endl;
        }
        return pass;
    }

    void run_kernels() {
        for (auto ctx : contexts) {
#ifdef DEBUG
            std::cout << "Running " << ctx->name << std::endl;
#endif

            ctx->init();
            std::vector<float> times;
            for (int i = 0; i < kernel_runs; ++i) {
                float t = ctx->run();
                times.push_back(t);
            }
            ctx->uninit();
            // avg/std dev/ min, max, med
            std::vector<float> timing_stats = stats_from_vec(times);
#ifdef DEBUG
            std::cout << "Actual runtimes:" << std::endl;
            for (int i=0; i < times.size(); ++i) {
                std::cout << std::setw(10) << times[i];
                if(i%10==9) {std::cout << std::endl;}
            }
            std::cout << std::endl ;

            int w = 15;
            std::cout << "Timing stats:" << std::endl;
            std::cout << std::setw(w) << "min"
                      << std::setw(w) << "med"
                      << std::setw(w) << "max"
                      << std::setw(w) << "avg"
                      << std::setw(w) << "stddev" << std::endl;
            for (auto v : timing_stats) {
                std::cout << std::setw(w) << v;
            }
            std::cout << std::endl << std::endl;
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
            std::cerr << "Not running kernels!" << std::endl;
        }
    }

    // output_file << "kernel_type,array_size,tpb,min,med,max,avg,stddev" << endl ;
    void write_data(kernel_ctx_t* ctx, std::vector<float> data) {
        std::stringstream s;
        copy(data.begin(),data.end()-1, std::ostream_iterator<float>(s,",")); // https://stackoverflow.com/questions/9277906/stdvector-to-string-with-custom-delimiter
        std::string wo_last_comma = s.str();
        wo_last_comma.pop_back(); // https://stackoverflow.com/questions/2310939/remove-last-character-from-c-string
        output_file << ctx->name << "," << ctx->N << "," << ctx->Bsz << "," << wo_last_comma << std::endl ;
    }
};
