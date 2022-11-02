#pragma once
/**
 * @file interleaved_copy.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from InterleavedCopyContext; testing for burst mode 
 * @version 0.1
 * @date 2022-02-24
 * 
 * Meant to ensure the blocks last the entire time on GPU, i.e. the 
 * blocks that are started on each SM persist throughout the lifetime
 * of the kernel. 
 * 
 */

#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

using std::string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it, int elements>
__forceinline__ __host__ __device__        
void interleaved_full_life_kernel(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    // unsigned long long b_idx = blockIdx.x;
    // unsigned long long t_idx = threadIdx.x;
    // unsigned long long Bsz = blockDim.x;
    // unsigned long long Gsz = gridDim.x;
    
    // int block_life = N / gridDim.x / elements; 
    unsigned long long start_idx = blockIdx.x * blockDim.x * elements + threadIdx.x;
    unsigned int cycle_offset = gridDim.x * blockDim.x * elements;

    for(int x=0; x < N / ( gridDim.x * blockDim.x * elements); ++x) {
        for(int y=0; y < elements; ++y) {
            unsigned long long data_idx =  start_idx + cycle_offset * x + blockDim.x*y;
            if(data_idx >= N) return;
            gpu_out[data_idx] = gpu_in[data_idx];
        }
    }
}


template<typename vt, typename it, int elements>
__global__        
void uncoalesced_reuse_kernel_for_regs(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        interleaved_full_life_kernel<vt, it, elements>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, int elements>
struct InterleavedCopyFullLifeContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        unsigned long long N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];

        int data_reads_per_element = 1;
        int index_reads_per_element = 0;
        int writes_per_element = 1;
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;
            unsigned long long N;

            __device__        
            void operator() (unsigned int idx){
                extern __shared__ int dummy[];
                interleaved_full_life_kernel<vt, it, elements>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        InterleavedCopyFullLifeContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            assert(N % (elements) == 0);
            this->name = "InterleavedCopyFullLife"; 

            int occupancy_blocks = int(this->get_occupancy() * float(this->dev_ctx->props_.maxThreadsPerMultiProcessor)) / this->Bsz;
            cout << "Occupancy = " << this->get_occupancy() << endl;
            this->Gsz = this->dev_ctx->props_.multiProcessorCount * occupancy_blocks;
            assert(this->Gsz > 0);
           
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~InterleavedCopyFullLifeContext(){}

        void init_inputs(bool& pass) override {
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {}

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
            ctx.N = N;
        }

        void output_config_info() override {
            cout << "InterleavedCopyFullLife with : "
                 << " Elements/cycle=" << elements 
                 << " Blocks used="<< this->Gsz 
                 << " occupancy=" << this->get_occupancy() <<  endl;
        }

        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        bool local_check_result() override {
            bool pass = true;
            unsigned long long i = 0;
            for(; i<N; ++i){
                if(in[i] != out[i]){
                    cout << "Validation Failed at " << i << ": in="<<in[i] << " out="<< out[i] << endl;
                    pass = false;
                    break;
                }
            }

            if(!pass) {
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(10) << "IN"<<"|" <<std::setw(10)<<"OUT " << endl; 
                int output_size = 10;
                unsigned long long j = max((int)0, (int)(i - output_size/2));
                for(int k=0; k < output_size; ++k, ++j) { 
                    cout << std::setw(10) << in[j] <<"|" <<std::setw(10)<<out[j] << endl; 
                }
            }
            return pass;
        }

        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, uncoalesced_reuse_kernel_for_regs<vt,it,elements>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
