#pragma once
/**
 * @file driver.h
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Class to drive testing of kernels, including verification
 * @version 0.1
 * @date 2022-02-04
 * 
 */

// #include <kernel_types.h>
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
using std::stringstream;
using std::setw;

/**
 * @brief Container for verifying, running, and saving the timing statistics 
 *      for a specific kernel workload
 * 
 * @tparam kernel_ctx_t Specific kernel context class to run
 */
template <typename kernel_ctx_t>
class MicrobenchmarkDriver {
   private:
    string output_filename_;            // Name of file to save output data to (CSV)
    ofstream output_file;               // File handle for output data to stream into
    bool output_file_started = false;   // Flag to determine whether header has been written
    device_context* dev_ctx_;           // References CUDA Device Context structure for device parameters
    vector<kernel_ctx_t*> contexts;     // Set of different kernel instances to test (just different occupancies at the moment)

    int N = 0;                          // Associated with workload size; often number of threads
    int bs = 0;                         // Block size (unused?)

   public:
    int kernel_runs = 25;               // Number of runs performed for timing statistics
    int kernel_checks = 1;              // Number of runs performed for kernel verification (defined in kernel context)

    /**
     * @brief Construct a new Microbenchmark Driver object
     * 
     * Sets up at least one, but possibly several, instances of `kernel_ctx_t`. 
     * Number and configuration depends on inputs. 
     * 
     * @param N Workload size
     * @param bs_vec Vector of block sizes to test over; currently only single value used 
     * @param output_filename Name to save timing statistics and other data to
     * @param dev_ctx Reference to CUDA GPU device context structure
     * @param span_occupancies Flag for whether to run kernel for valid relative occupancies in range (0,1] 
     */
    MicrobenchmarkDriver(int N, vector<int>& bs_vec, string output_filename, device_context* dev_ctx, bool span_occupancies=false) :
        output_filename_(output_filename) {
        //dev_ctx->init(); // assumed ctx is initialized already (why init in every single driver?)
        dev_ctx_ = dev_ctx;

        for (int bs : bs_vec) {
            kernel_ctx_t* curr_ctx = new kernel_ctx_t(N, bs, dev_ctx);
            curr_ctx->print_register_usage();
            if(span_occupancies && curr_ctx->okay) {
                vector<int> shdmem_allocs = curr_ctx->shared_memory_allocations();
#ifdef DEBUG
                cout << "Valid ShdMem alloc amounts for "<< curr_ctx->name <<": ";
                for(int x : shdmem_allocs) {cout << " " << x;}
                cout << endl;
#endif
                if(shdmem_allocs.size() > 0) {
                    for(int i=0; i < shdmem_allocs.size(); ++i){
                        contexts.push_back(new kernel_ctx_t(N, bs, dev_ctx, shdmem_allocs[i]));
                    }
                }
            }
            contexts.push_back(curr_ctx);
        }
    }
    ~MicrobenchmarkDriver() {
        for (auto ctx : contexts) {
            if (ctx)
                delete ctx;
        }
        output_file.close();
    }

    void set_config_bool(bool val) {
        for (auto ctx : contexts) {
            ctx->set_config_bool(val);
        }
    }

    /**
     * @brief Test each kernel for accuracy (only 1 now)
     * 
     * @return true All kernels passed check
     * @return false One kernel failed check 
     */
    bool check_kernels() {
        bool pass = true;
#ifdef DEBUG
        cout << "Beginning check" << endl;
#endif
        // for (auto ctx : contexts) {
            auto ctx = contexts.back();
            if(!ctx->okay) {return false;}
#ifdef DEBUG
            ctx->output_config_info();
#endif
            if(!ctx->init()) {return false;}
            for (int i = 0; i < kernel_checks; ++i) {
                pass = (pass && ctx->run_and_check());
                if (!pass) break;
            }
            ctx->uninit();
            // if (!pass) break;
        // }
        if (!pass) {
            cerr << "Kernel failed check!" << endl;
        } else {
#ifdef DEBUG
            cout << "Kernel passed check!" << endl;
#endif
        }
        return pass;
    }

    /**
     * @brief Perform `kernel_runs` runs of each kernel, then save statistical data
     * 
     */
    void run_kernels() {
#ifdef DEBUG
            cout << "Beginning actual runs" << endl;
#endif
        for (auto ctx : contexts) {
            if(!ctx->okay) {return;}
#ifdef DEBUG
            ctx->output_config_info();
#endif
            if(!ctx->init()) {return;}
            using timing_precision_t = double;
            vector<timing_precision_t> times;
            for (int i = 0; i < kernel_runs; ++i) {
                timing_precision_t t = ctx->run();
                times.push_back(t);
            }
            ctx->uninit();
            // avg/std dev/ min, max, med
            vector<timing_precision_t> timing_stats = stats_from_vec(times);
            timing_precision_t throughput =  (timing_precision_t) ctx->get_total_bytes_processed() / (1024*1024*1024)  // Total data processed * 1 GB/(1024^3) B
                    / timing_stats[0]                                                       // Min time to finish (1/ms)
                    * 1000;                                                                 // 1000 ms / 1 s
            timing_stats.push_back(throughput);
            timing_precision_t fraction_theoretical_bw_achieved = throughput / dev_ctx_->theoretical_bw_; 
            timing_stats.push_back(fraction_theoretical_bw_achieved);

#ifdef DEBUG
#ifdef DEBUG_LEVEL1
            cout << "Actual runtimes:" << endl;
            for (int i = 0; i < times.size(); ++i) {
                cout << setw(10) << times[i];
                if (i % 10 == 9) {
                    cout << endl;
                }
            }
            cout << endl;
#endif

            int w = 15;
            cout << "Timing stats:" << endl;
            cout << setw(w) << "min"
                      << setw(w) << "med"
                      << setw(w) << "max"
                      << setw(w) << "avg"
                      << setw(w) << "stddev"
                      << setw(w) << "througput" 
                      << setw(w) << "frac mx bw" << endl;
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

    /**
     * @brief Wrapper for check_kernels() and run_kernels()
     * 
     * First, checks kernels; if pass checks, runs kernels. 
     * 
     * @return true 
     * @return false 
     */
    bool check_then_run_kernels() {
        bool pass = check_kernels();
        if (pass) {
            run_kernels();
        } else {
            cerr << "Not running kernels!" << endl;
        }
        return pass; 
    }

    /**
     * @brief Add headers to output file
     * 
     */
    void start_output_file(){
        // Guard against overwriting data
        struct stat output_file_buffer;
        int i = 0;
        string original_filename = output_filename_;
        while (stat(output_filename_.c_str(), &output_file_buffer) == 0)
        {
            cerr << "The file '" << output_filename_ << "' already exists in the output directory!" << endl;
            ++i;
            output_filename_ = original_filename+"("+to_string(i)+")";
        }

        // output_file << "Array_size,tpb,ept,bwss,twss,num_blocks,fraction_of_l2_used_per_block,num_repeat,theoretical_bandwidth"
        //              << ",shuffle_type,kernel_type,blocks_per_sm,min,med,max,avg,stddev,achieved_throughput" << endl ;
        output_file.open(output_filename_.c_str());
        output_file << "kernel_type,array_size,value_size,index_size,tpb,occupancy,min,med,max,avg,stddev,throughput,fraction_of_max_bandwidth" ;
        if(contexts[0]->get_extra_config_parameters().compare("") != 0) {
            output_file << "," << contexts[0]->get_extra_config_parameters() ;
        }
        output_file << endl;
        output_file_started = true;

    }

    // output_file << "kernel_type,array_size,tpb,occupancy,min,med,max,avg,stddev" << endl ;
    template<typename T>
    void write_data(kernel_ctx_t* ctx, vector<T> data) {
        if(!output_file_started) { start_output_file(); }
        stringstream s;
        copy(data.begin(), data.end(), std::ostream_iterator<T>(s, ","));  // https://stackoverflow.com/questions/9277906/stdvector-to-string-with-custom-delimiter
        string wo_last_comma = s.str();
        wo_last_comma.pop_back();  // https://stackoverflow.com/questions/2310939/remove-last-character-from-c-string
        output_file << ctx->name << "," << ctx->N << "," << ctx->vt_size << "," << ctx->it_size << "," << ctx->Bsz << ","  << ctx->get_occupancy() << "," << wo_last_comma;
        if(ctx->get_extra_config_parameters().compare("") != 0) {
            output_file << "," << ctx->get_extra_config_values();
        }
        output_file << endl;
    }

    /**
     * @brief Get the number of context in this driver instance
     * 
     * @return int Number of kernel contexts (configurations)
     */
    int get_num_contexts() {
        return (int)contexts.size();
    }

    /**
     * @brief Get the number of total runs performed in this driver instance
     * 
     * @return int Total number of kernel runs in the driver (# kernels * kernel_runs)
     */
    int get_total_runs() {
        return get_num_contexts() * kernel_runs;
    }
};
