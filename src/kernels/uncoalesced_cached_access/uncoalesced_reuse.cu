#pragma once
/**
 * @file uncoalesced_reuse.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from InterleavedCopyContext; testing for burst mode 
 * @version 0.1
 * @date 2022-02-28
 * 
 * Meant to evaluate the effect coalescing/non-coalescing has on accesses 
 * to cached data (data within the memory hierarchy).
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

#define ELEMENTS_REUSED 4

template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts>
__forceinline__ __host__ __device__        
void uncoalesced_reuse_kernel(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    uint b_idx = blockIdx.x;
    uint t_idx = threadIdx.x;
    uint Bsz = blockDim.x;
    uint Gsz = gridDim.x;

    uint num_warps = Bsz / 32;
    
    // Preload data
    if constexpr(preload_for_reuse) {
        vt tmp = gpu_in[idx];
        if(tmp < 0) return; // all values should be > 0; this is just to ensure this write is not removed
    }

    int start_idx = b_idx * Bsz;
    int generated_indices[ELEMENTS_REUSED];
    for(int i=0; i<ELEMENTS_REUSED; ++i){
        if constexpr(!avoid_bank_conflicts) {
            int tmp_t_idx = (t_idx+i) % Bsz;
            generated_indices[i] = ( tmp_t_idx % num_warps) * 32 + tmp_t_idx / num_warps + start_idx;
        } else {
            generated_indices[i] = ( (tmp_t_idx % 32) * 32 + (tmp_t_idx % 32 + j / num_warps ) % 32) % block_size + start_idx; 
        }
    }

    vt output_val = 0;
    for(int i=0; i<ELEMENTS_REUSED; ++i) {
        output_val += gpu_in[generated_indices[i]];
    }

    gpu_out[idx] = output_val;
}


template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts>
__global__        
void kernel_for_regs(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        uncoalesced_reuse_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, bool preload_for_reuse, bool avoid_bank_conflicts>
struct UncoalescedReuseContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
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
            void operator() (uint idx){
                extern __shared__ int dummy[];
                uncoalesced_reuse_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        UncoalescedReuseContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            // assert(N % (local_group_size * elements * block_life) == 0);
            this->name = "UncoalescedReuse"; 
            // this->Gsz /= local_group_size * elements * block_life;
            assert(this->Gsz > 0);
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~UncoalescedReuseContext(){}

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
            cout << this->name << " with : "
                 <<" preloading?=" << preload_for_reuse 
                 << " avoiding bank conflicts?=" << avoid_bank_conflicts << endl;
        }

        void local_execute() override {
            if(this->dev_ctx->props_.major >= 7) {
                cudaFuncSetAttribute(compute_kernel<gpu_ctx>, cudaFuncAttributeMaxDynamicSharedMemorySize, this->dev_ctx->props_.sharedMemPerMultiprocessor);
            }
            compute_kernel<gpu_ctx><<<Gsz, Bsz, this->shared_memory_usage>>>(N, ctx);
            cudaPrintLastError();
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
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel_for_regs<vt,it,preload_for_reuse,avoid_bank_conflicts>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
