// driver.cpp
// simple driver file for kernel testing

#include <kernel_types.h>
#include <local_cuda_utils.h>
#include <stats.h>

#include <fstream>
#include <iostream>
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
    int kernel_runs = 50;
    int kernel_checks = 2;

   public:
    MicrobenchmarkDriver(int N, std::vector<int> bs_vec) {
        for (int bs : bs_vec) {
            contexts.push_back(new kernel_ctx_t(N, bs));
        }
    }
    ~MicrobenchmarkDriver() {
        for (auto ctx : contexts) {
            if (ctx)
                delete ctx;
        }
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
            // avg/std dev/ min, max, med
            // output to file
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

};
