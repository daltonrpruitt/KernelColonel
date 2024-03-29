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
void uncoalesced_reuse_kernel(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    unsigned int b_idx = blockIdx.x;
    unsigned int t_idx = threadIdx.x;
    unsigned int Bsz = blockDim.x;
    // unsigned int Gsz = gridDim.x;

    unsigned int num_warps = Bsz / 32;
    
    // Preload data
    if constexpr(preload_for_reuse) {
        vt tmp = gpu_in[idx];
        if(tmp < 0) return; // all values should be > 0; this is just to ensure this write is not removed
    }

    int start_idx = b_idx * Bsz;
    int generated_indices[ELEMENTS_REUSED];
    for(int i=0; i<ELEMENTS_REUSED; ++i){
        int tmp_t_idx = (t_idx+i) % Bsz;
        if constexpr(!avoid_bank_conflicts) {
            generated_indices[i] = ( tmp_t_idx % num_warps) * 32 + tmp_t_idx / num_warps + start_idx;
        } else {
            generated_indices[i] = ( (tmp_t_idx % 32) * 32 + (tmp_t_idx % 32 + tmp_t_idx / num_warps ) % 32) % Bsz + start_idx; 
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
void kernel_for_regs(unsigned int idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
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

        int data_reads_per_element = ELEMENTS_REUSED;
        int index_reads_per_element = 0;
        int writes_per_element = 1;
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;
            unsigned long long N;

            __device__        
            void operator() (unsigned int idx){
                extern __shared__ int dummy[];
                uncoalesced_reuse_kernel<vt, it, preload_for_reuse, avoid_bank_conflicts>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        UncoalescedReuseContext(int n, int bs, GpuDeviceContext* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            // assert(N % (local_group_size * elements * block_life) == 0);
            this->name = "UncoalescedReuse"; 
            // this->Gsz /= local_group_size * elements * block_life;
            assert(this->Gsz > 0);
            if constexpr(preload_for_reuse) {
                data_reads_per_element += 1;
            }

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
                 << " avoiding bank conflicts?=" << avoid_bank_conflicts
                 << " occupancy=" << this->get_occupancy() <<  endl;
        }

        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        bool local_check_result() override {
            bool pass = true;
            unsigned long long i = 0;
            int num_warps = Bsz / 32;
            for(int i=0; i < this->Gsz; ++i) {
                int start_idx = i * Bsz;
                for (int j=0; j < Bsz; ++j){
                    unsigned int global_tidx = start_idx + j;
                    vt sum = 0;
                    for(int e=0; e<ELEMENTS_REUSED; ++e) {
                        int tmp_t_idx = (j + e) % Bsz;
                        if constexpr(!avoid_bank_conflicts) {
                            sum += in[( tmp_t_idx % num_warps) * 32 + tmp_t_idx / num_warps + start_idx];
                        } else {
                            sum += in[( (tmp_t_idx % 32) * 32 + (tmp_t_idx % 32 + tmp_t_idx / num_warps ) % 32) % Bsz + start_idx];

                        }
                    }
                    if (out[global_tidx] != sum) {
                        cout << "Validation Failed at " << global_tidx << ": in="<<in[global_tidx] << " out="<< out[global_tidx] << endl;
                        pass = false;
                        break;
                    }
                }
                if(!pass) break;
            }

            if(!pass) {
                cout << "Debug dump of in and out array: " << endl;
                cout << std::setw(10) << "IN" << "  |" << std::setw(10) << "OUT " << endl; 
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
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, kernel_for_regs<vt,it,preload_for_reuse,avoid_bank_conflicts>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
