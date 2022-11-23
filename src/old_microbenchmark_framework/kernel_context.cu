#pragma once 
/**
 * @file kernel_context.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides context information for GPU kernel execution of driver
 * @version 0.1
 * @date 2022-01-27
 * 
 */

#include <local_cuda_utils.h>
#include <device_props.h>

#include <vector>
#include <algorithm>

#include <cuda.h>

using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;


/**
 * @brief Simple wrapper to facilitate automatic kernel execution
 * 
 * @tparam kernel_ctx_t Type of the gpu device kernel context
 * @param N Number of threads should run
 * @param ctx The actual device kernel context information (subset of cpu KernelContext)
 */
template<typename kernel_ctx_t>
__global__
void compute_kernel(unsigned long long N, kernel_ctx_t ctx) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    ctx(idx);
}

/**
 * @brief Reusable way to setup and execute kernels for any of the derived context classes
 * 
 * Like most of these classes, needs to be reworked...
 * 
 * @tparam kernel_ctx_t 
 * @param N Size of relevant param, typically array size/number of threads (varies)
 * @param Gsz Grid size
 * @param Bsz Thread block size
 * @param shdmem_usage Amount of shared memory allocated per thread block 
 * @param dev_ctx Device context information (CUDA-supplied)
 * @param ctx GPU kernel context 
 * @return float Execution time of kernel
 */
template<typename kernel_ctx_t>
inline
float local_execute_template(int N, int Gsz, int Bsz, int shdmem_usage, GpuDeviceContext* dev_ctx, kernel_ctx_t ctx) {
    if(dev_ctx->props_.major >= 7) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, compute_kernel<kernel_ctx_t>);
	
        int shmem = dev_ctx->props_.sharedMemPerMultiprocessor-attr.sharedSizeBytes;
        if(dev_ctx->props_.major == 8) { shmem -= 1024; }
	
        cudaFuncSetAttribute(compute_kernel<kernel_ctx_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
        cudaFuncSetAttribute(compute_kernel<kernel_ctx_t>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaPrintLastError();
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    compute_kernel<kernel_ctx_t><<<Gsz, Bsz, shdmem_usage>>>(N, ctx);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaPrintLastError();

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return time; 
}

/**
 * @brief Controller of kernel-specific execution information (setup included)
 * 
 * Went from a simple CPU-side collection of information needed to execute a kernel
 * to being the actual kernel execution engine, basically. 
 * Needs to be broken down based on functionality (imo).
 * 
 * @tparam vt Value type (data arrays)
 * @tparam it Index type (indirection arrays)
 */
template<typename vt, typename it>
struct KernelCPUContext {
    public:
        unsigned int vt_size = sizeof(vt);
        unsigned int it_size = sizeof(it);
        typedef it IT;
        string name;
        unsigned long long N=0;
        unsigned long long input_size=0;
        unsigned long long output_size=0;
        unsigned long long indices_size=0;
        int Bsz=-1;
        int Gsz=-1;
        int num_in_data=-1;
        int num_out_data=-1;
        int num_total_data=-1;
        int num_indices=-1;
        
        bool okay = true;
        bool initialized = false;

        size_t shared_memory_usage=0;
        int register_usage=-1;
        int max_blocks_simultaneous_per_sm=-1;

        int total_data_reads;
        int total_index_reads;
        int total_writes;

        KernelCPUContext(int bs, int shd_mem_alloc=0)
            : N(n), Bsz(bs), Gsz( (n+bs-1)/bs ), dev_ctx(d_ctx), shared_memory_usage(shd_mem_alloc) {}

        ~KernelCPUContext() = default;

        /**
         * @brief Set the config bool object
         * 
         * Was required to automatically scale the occupancy with the ILP in one of the kernels. 
         * If something like this is ever required again, this type of stuff should be 
         * accomplished through config files.
         * 
         * @param val Value to set the config_bool to
         */
        // virtual void set_config_bool(bool val) {
        //     cerr << "set_config_bool() is undefined for this kernel!" << endl;
        // };

        /**
         * @brief Base output of configuration information
         * 
         * To be overridden if want to show more info. 
         */
        virtual void output_config_info() {
            cout << name << endl; 
        }

        /**
         * @brief Set the device ptrs to correct GPU memory objects
         * 
         */
        virtual void set_dev_ptrs() {}

        /**
         * @brief Ensure kernel definer makes this function (or uses template)
         * 
         * @return float Time of kernel execution
         */
        virtual float local_execute() = 0;

        /**
         * @brief Outer wrapper for kernel execution
         * 
         * @return float Time of kernel execution (-1 if failed)
         */
        float execute() {
            if(!okay) return -1.0;

            float time = local_execute();

            return time;
        }

        /**
         * @brief Ensure kernel definer makes this function
         * 
         * @return bool Is resulting state correct?
         */
        virtual bool local_check_result() = 0;

        /**
         * @brief Outer wrapper for checking
         * 
         * Handles some of the boilerplate for getting ready to check. 
         * 
         * @return true Passed check
         * @return false Failed check or previous failure
         */
        bool check_result() {
            if(!okay){
                cout << "Cannot check "<< name << " due to previous failure!" << endl;
                return false;
            };

            bool pass = true;
            for(int i=num_in_data; i < num_total_data; ++i) {
                cudaErrChk(cudaMemcpy(host_data[i].data(), device_data_ptrs[i], output_size * sizeof(vt), cudaMemcpyDeviceToHost),"copying device_data_ptrs["+to_string(i)+"] to host_data["+to_string(i)+"]", pass);
            }            
            if(!pass) {free(); okay = false;}

            return local_check_result();
        }

        /**
         * @brief Ensure is ready to execute, then execute
         * 
         * @return float Kernel execution timing
         */
        float run() {
            if(!initialized) {
                if(!init()) return -1.0;
            }
            return execute();
        }
        

        /**
         * @brief Run kernel, then check results
         * 
         * @return true Passed check
         * @return false Failed during running or failed check
         */
        bool run_and_check() {
            run(); // ignore time
            return check_result();     
        }

    /**
     * @brief Ensure kernel definer makes this
     * 
     * This is used in computing occupancy, and requires references to the global kernel functions. 
     * 
     * @param pass Executed okay?
     */
    virtual void local_compute_register_usage(bool& pass) = 0;

    /**
     * @brief Compute the number of blocks that can execute on a single SM at once
     * 
     * SM = Streaming Multiprocessor
     * 
     * @param pass Computed okay?
     */
    void compute_max_simultaneous_blocks(bool& pass) {
        local_compute_register_usage(pass);
        if(!pass) { okay = false; return;}
        int due_to_block_size = (int) floor(dev_ctx->props_.maxThreadsPerMultiProcessor / Bsz); 
        int due_to_registers =  (int) floor(dev_ctx->props_.regsPerMultiprocessor / (register_usage * Bsz));
        max_blocks_simultaneous_per_sm = std::min({due_to_block_size, 
                                            due_to_registers, dev_ctx->props_.maxBlocksPerMultiProcessor});

    }

    /**
     * @brief Compute the shared memory amounts needed to achieve a range of occupancies
     * 
     * The occupancies are typically the powers of 1/2: 1, 1/2, 1/4, 1/8, 1/16, etc. 
     * until the smallest number of blocks an SM can execute (typically 1 or 2 depending on architecure). 
     * For example, on the V100, the fewest blocks that can execute is 2. With a block size of 64, and the maximum
     * threads per SM value of 2048, this gives an occupancy of 2*64/2048 = 0.0625 (1/16)
     * So, our occupancies tested would be 0.0625, 0.125, 0.25, 0.5, and 1.0, assuming no other contributing factors 
     * (such as high register usage within the kernel). 
     * 
     * @return vector<int> The list of shared memory allocations for the given kernel and configuration.
     */
    vector<int> shared_memory_allocations() {
        vector<int> alloc_amounts; 
        bool pass = true;
        if(max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if(!pass) { 
            okay = false;  
            alloc_amounts.push_back(-1);
            return alloc_amounts;
        }
        int max_shd_mem_per_block = dev_ctx->props_.sharedMemPerBlock;
        int max_shd_mem_per_proc = dev_ctx->props_.sharedMemPerMultiprocessor;
        if(dev_ctx->props_.major == 8) {
            max_shd_mem_per_block = 164 * 1024;
            max_shd_mem_per_proc =  164 * 1024;
        }

        int min_blocks_due_to_shd_mem = max_shd_mem_per_proc / max_shd_mem_per_block;

        for(int i=min_blocks_due_to_shd_mem; i < max_blocks_simultaneous_per_sm ; i*=2) {
            int sm_alloc = std::min((max_shd_mem_per_proc / i ) / 256 * 256, max_shd_mem_per_block);
            if(dev_ctx->props_.major == 8) {sm_alloc -= 1024;}
            if ( std::find(alloc_amounts.begin(), alloc_amounts.end(), sm_alloc) == alloc_amounts.end() ) {
                alloc_amounts.push_back(sm_alloc);
            }
        }
        return alloc_amounts;
    }

    /**
     * @brief Compute the occupancy of the current kernel and configuration
     * 
     * @return float Relative occupancy (0-1)
     */
    float get_occupancy() {
        bool pass = true;
        if(max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if(!pass) { 
            okay = false;  
            return -1.0;
        }

        int max_blocks_shared_mem;
        if(shared_memory_usage == 0) {
            max_blocks_shared_mem = dev_ctx->props_.maxBlocksPerMultiProcessor;
        } else {
            if(dev_ctx->props_.major == 8) {
                max_blocks_shared_mem = dev_ctx->props_.sharedMemPerMultiprocessor / (shared_memory_usage+1024);
            } else {
                max_blocks_shared_mem = dev_ctx->props_.sharedMemPerMultiprocessor / shared_memory_usage;
            }
        }

        int max_blocks_simul = std::min(max_blocks_simultaneous_per_sm, max_blocks_shared_mem);
        int num_threads_simul = max_blocks_simul * Bsz; 
        return float(num_threads_simul) / float(dev_ctx->props_.maxThreadsPerMultiProcessor);
    }

    /**
     * @brief Compute the shared memory needed to achieve the specified occupancy
     * 
     * @param occupancy Occupancy value to achieve
     * @return int Shared memory required to be allocated to blocks
     */
    int get_sharedmemory_from_occupancy(float occupancy) {
        bool pass = true;
        if(max_blocks_simultaneous_per_sm < 0) compute_max_simultaneous_blocks(pass);
        if(!pass) { 
            okay = false;  
            return -1;
        }

        int blocks = float(dev_ctx->props_.maxThreadsPerMultiProcessor / Bsz) * occupancy;
        if(blocks > max_blocks_simultaneous_per_sm){
            cerr << "Try to get occupancy higher than architecture allows!" << endl;
            return -1;
        }

        int max_shd_mem_per_block = dev_ctx->props_.sharedMemPerBlock;
        int max_shd_mem_per_proc = dev_ctx->props_.sharedMemPerMultiprocessor;
        // if(dev_ctx->props_.major == 8) {max_shd_mem_per_block = max_shd_mem_per_proc;}
        if(dev_ctx->props_.major == 8) {
            max_shd_mem_per_block = 164 * 1024;
            max_shd_mem_per_proc =  164 * 1024;
        }


        int shdmem = max_shd_mem_per_proc / blocks; 

        if(shdmem > max_shd_mem_per_block) {
            cerr << "Cannot set shared memory high enough to match occupancy of " << occupancy <<"!" << endl;
            shdmem = max_shd_mem_per_block;
        }
        if(dev_ctx->props_.major == 8) {shdmem -= 1024;}
        return shdmem;
    }

    /**
     * @brief Debug print of register usage for the kernel
     * 
     */
    void print_register_usage() {
        bool pass = true; 
        if(register_usage < 0) { 
            local_compute_register_usage(pass);
        }
        if(!pass) {cerr << "Cannot get register usage for " << name << "!" << endl;}
        else { cout << name << " register usage = " << register_usage << endl;}
    }

    /**
     * @brief Compute the total bytes processed (global memory) from user-specifications
     * 
     * This has proven correct so far, but needs to be revisited in the rework. 
     * I would prefer to have this computed from some kind of better analysis of the kernel. 
     * Maybe just as user input in a config file?
     * 
     * @return unsigned long long  Bytes of global memory processed across kernel execution
     */
    unsigned long long get_total_bytes_processed() {
        return ( total_data_reads+ total_writes)*sizeof(vt) +  total_index_reads*sizeof(it);
    }

    /**
     * @brief Allow user-defined extra configuration data (headers)
     * 
     * @return string Comma-separated configuration headers to add to output csv data file
     */
    virtual string get_extra_config_parameters() { return "";}

    /**
     * @brief Allow user-defined extra configuration data (data)
     * 
     * @return string Comma-separated configuration data to add to output csv data file
     */
    virtual string get_extra_config_values() { return "Error!";}
};
