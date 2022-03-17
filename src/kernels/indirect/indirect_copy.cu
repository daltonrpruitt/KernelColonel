#pragma once
/**
 * @file indirect_copy.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-03-16
 * 
 * Based on simple_indirection.cu 
 * Meant to be used to test various access patterns in an indirect manner. 
 * This allows for the pattern to be discerned at runtime to reflect actual 
 * irregular application conditions.
 */

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>
#include <indices_generation.h>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it, int ILP>
__forceinline__ __host__ __device__        
void kernel_indirect_copy(uint idx, vt* in, vt* out, it* indices){
    it indxs[ILP];
    uint local_start_idx = idx + blockIdx.x * blockDim.x * (ILP-1);
    for(int i=0; i < ILP; ++i) {
        indxs[i] = indices[local_start_idx + blockDim.x*i];
    }

    vt arr[ILP];
    for(int i=0; i < ILP; ++i) {
        arr[i] = in[indxs[i]];
    }

    for(int i=0; i < ILP; ++i) {
        out[local_start_idx + blockDim.x*i] = arr[i];
    }
}

template<typename vt, typename it, int ILP>
__global__        
void kernel_for_regs_indirect_copy(uint idx, vt* in, vt* out, it* indices){
    extern __shared__ int dummy[];
    kernel_indirect_copy<vt, it, ILP>(idx, in, out, indices);
}

template<typename vt, typename it, int shuffle_size, int ILP, indices_pattern idx_pattern>
struct IndirectCopyContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        int N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        // Setup inputs/outputs, both data and indicies
        // Can be still in vector form, but is easier to identify explicitly with names   
        // Example
        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vector<it> & indices = super::host_indices[0];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];
        it* & d_indices = super::device_indices_ptrs[0];
        

        int data_reads_per_element = 1; // Actual number 
        int writes_per_element = 1; // Actual number
        int index_reads_per_element = 1; // Actual number

        struct gpu_ctx {
            vt * gpu_in; 
            vt * gpu_out;
            it * gpu_indices;

            __device__        
            void operator() (uint idx){
                extern __shared__ int dummy[];
                kernel_indirect_copy<vt, it, ILP>(idx, gpu_in, gpu_out, gpu_indices);
            }
        } ctx ;

        IndirectCopyContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 1, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "IndirectCopy";

            this->Gsz /= ILP;
            assert(this->Gsz > 0);

            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~IndirectCopyContext(){}

        void init_inputs(bool& pass) override {
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {
            bool debug = false;
// #ifdef DEBUG
//             debug = true;
// #endif
            indices.reserve(N);
            if( index_patterns[idx_pattern](indices.data(), N, Bsz, shuffle_size, debug) != 0) {
                cerr << "Failed to generate indices!"; 
                this->okay = false;
            }
        }

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
            ctx.gpu_indices = d_indices;
        }

        bool local_check_result() override {
            for(int i=0; i<N; ++i){
                if(in[indices[i]] != out[i]){
                    cout << "Validation Failed at " << i << ": in="<<in[i] << " out="<< out[i] << endl;
                    return false;
                }
            }
            return true;
        }

        // No change
        float local_execute() override {
            return local_execute_template<gpu_ctx>(N, Gsz, Bsz, this->shared_memory_usage, this->dev_ctx, ctx);
        }

        // No change
        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel_for_regs_indirect_copy<vt,it,ILP>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
