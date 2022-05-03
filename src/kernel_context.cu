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


template<typename kernel_ctx_t>
__global__
void compute_kernel(unsigned long long N, kernel_ctx_t ctx) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    ctx(idx);
}

template<typename kernel_ctx_t>
inline
float local_execute_template(int N, int Gsz, int Bsz, int shdmem_usage, device_context* dev_ctx, kernel_ctx_t ctx) {
    if(dev_ctx->props_.major >= 7) {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, compute_kernel<kernel_ctx_t>);
        int shmem = dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
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


template<typename vt, typename it>
struct KernelCPUContext {
    public:
        uint vt_size = sizeof(vt);
        uint it_size = sizeof(it);
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

        device_context* dev_ctx;

        vector<vector<vt>> host_data{(unsigned long)num_total_data};
        vector<vt *> device_data_ptrs{(unsigned long)num_total_data};

        
        vector<vector<it>> host_indices{(unsigned long)num_indices};
        vector<it *> device_indices_ptrs{(unsigned long)num_indices};


        void free(){
            for(vt* ptr : device_data_ptrs)     { cudaFree(ptr); ptr = nullptr; }
            for(it* ptr : device_indices_ptrs)  { cudaFree(ptr); ptr = nullptr; }
        }
        
        void uninit() {
            if(!initialized) {return;}
            free();
            for(int i=0; i<num_total_data; ++i) { 
                vector<vt>().swap(host_data[i]); 
            }
            for(int i=0; i<num_indices; ++i) { 
                vector<it>().swap(host_indices[i]); 
            }
            initialized = false;
        }

        virtual void init_inputs(bool& pass) {};
        virtual void init_indices(bool& pass) {};

        KernelCPUContext(int in, int out, int indices, unsigned long long n, int bs, device_context* d_ctx, int shd_mem_alloc=0)
            : num_in_data(in), num_out_data(out), num_indices(indices), 
            num_total_data(in+out), N(n), Bsz(bs), Gsz( (n+bs-1)/bs ), dev_ctx(d_ctx), shared_memory_usage(shd_mem_alloc) {
            }

        bool init(){
            if(initialized) { return true; }
            bool pass = true;
            if(input_size == 0) { input_size = N; }
            if(output_size == 0) { output_size = N; }
            if(indices_size == 0) { indices_size = N; }

            compute_max_simultaneous_blocks(pass);
            if(pass) init_inputs(pass);
            if(pass) init_indices(pass);

            if(pass){

                device_data_ptrs.resize(num_total_data);

                for(int i=0; i < num_in_data; ++i) {
                    cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], input_size * sizeof(vt)),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                    if(!pass) break;
                }

                if(pass) {
                    for(int i=num_in_data; i < num_total_data; ++i) {
                        cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], output_size * sizeof(vt)),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                        if(!pass) break;
                    }
                }
                
                if(pass) {
                    for(int i=0; i < num_in_data; ++i) {
                        cudaErrChk(cudaMemcpy(device_data_ptrs[i], host_data[i].data(), input_size * sizeof(vt), cudaMemcpyHostToDevice), "copy host_data["+to_string(i)+"] to device_data_ptrs["+to_string(i)+"]", pass);                
                        if(!pass) break;
                    }
                }

                for(int i=0; i < num_indices; ++i) {
                    cudaErrChk(cudaMalloc((void **)&device_indices_ptrs[i], indices_size * sizeof(it)),"device_indices_ptrs["+to_string(i)+"] mem allocation", pass);
                    if(!pass) break;
                }

                if(pass) {
                    for(int i=0; i < num_indices; ++i) {
                        cudaErrChk(cudaMemcpy(device_indices_ptrs[i], host_indices[i].data(), indices_size * sizeof(it), cudaMemcpyHostToDevice), "copy host_indices["+to_string(i)+"] to device_indices_ptrs["+to_string(i)+"]", pass);                
                        if(!pass) break;
                    }
                }

                if(pass) { set_dev_ptrs(); }
            }

            if(!pass) {
                free(); 
                okay = false;
                cerr<<"Error in initializing "<<this->name << "for N="<<this->N<<" Bsz="<<this->Bsz;
                if(input_size != 0) cout << " input_sz="<<input_size;
                if(output_size != 0) cout << " output_sz="<<output_size;
                if(indices_size != 0) cout << " indices_sz="<<indices_size;
                cerr << " !" << endl;
            }
            if(pass) initialized = true;
            return pass;
        }

        ~KernelCPUContext(){
            uninit();            
        }

        virtual void set_config_bool(bool val) {
            cerr << "set_config_bool() is undefined for this kernel!" << endl;
        };

        virtual void output_config_info() {
            cout << name << endl; 
        }

        virtual void set_dev_ptrs() {}

        virtual float local_execute() = 0;

        float execute() {
            if(!okay) return -1.0;

            float time = local_execute();

            return time;
        }

        virtual bool local_check_result() = 0;

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

        float run() {
            if(!initialized) {
                if(!init()) return -1.0;
            }
            return execute();
        }
        
        bool run_and_check() {
            run(); // ignore time
            return check_result();     
        }

    virtual void local_compute_register_usage(bool& pass) = 0;

    void compute_max_simultaneous_blocks(bool& pass) {
        local_compute_register_usage(pass);
        if(!pass) { okay = false; return;}
        int due_to_block_size = (int) floor(dev_ctx->props_.maxThreadsPerMultiProcessor / Bsz); 
        int due_to_registers =  (int) floor(dev_ctx->props_.regsPerMultiprocessor / (register_usage * Bsz));
        max_blocks_simultaneous_per_sm = std::min({due_to_block_size, 
                                            due_to_registers, dev_ctx->props_.maxBlocksPerMultiProcessor});

    }

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
        if(dev_ctx->props_.major == 8) {max_shd_mem_per_block = max_shd_mem_per_proc;}
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
            max_blocks_shared_mem = dev_ctx->props_.sharedMemPerMultiprocessor / shared_memory_usage;
        }
        int max_blocks_simul = std::min(max_blocks_simultaneous_per_sm, max_blocks_shared_mem);
        int num_threads_simul = max_blocks_simul * Bsz; 
        return float(num_threads_simul) / float(dev_ctx->props_.maxThreadsPerMultiProcessor);
    }

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
        if(dev_ctx->props_.major == 8) {max_shd_mem_per_block = max_shd_mem_per_proc;}

        int shdmem = max_shd_mem_per_proc / blocks; 

        if(shdmem > max_shd_mem_per_block) {
            cerr << "Cannot set shared memory high enough to match occupancy of " << occupancy <<"!" << endl;
            shdmem = max_shd_mem_per_block;
        }
        if(dev_ctx->props_.major == 8) {shdmem -= 1024;}
        return shdmem;
    }

    void print_register_usage() {
        bool pass = true; 
        if(register_usage < 0) { 
            local_compute_register_usage(pass);
        }
        if(!pass) {cerr << "Cannot get register usage for " << name << "!" << endl;}
        else { cout << name << " register usage = " << register_usage << endl;}
    }

    unsigned long long get_total_bytes_processed() {
        return ( total_data_reads+ total_writes)*sizeof(vt) +  total_index_reads*sizeof(it);
    }

    virtual string get_extra_config_parameters() { return "";}
    virtual string get_extra_config_values() { return "Error!";}
};
