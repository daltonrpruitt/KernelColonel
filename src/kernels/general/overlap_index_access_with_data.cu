#pragma once
/**
 * @file overlap_index_access_with_data.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Frontload some number of indices accesses before accessing data
 * @version 0.1
 * @date 2022-02-14
 * 
 * Test latency amortization technique of frontloading indirection accesses
 * Exploiting ILP 
 * 
 */

#include <iostream>
#include <vector>
#include <string>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;


template<typename vt, typename it, int num_idxs>
__forceinline__ __host__ __device__        
void kernel(unsigned int idx, vt* in, vt* out, it* indices){
    unsigned int indir_idxs[num_idxs];
    
    int tidx = threadIdx.x;
    int Bidx = blockIdx.x;

    int Bsz = blockDim.x;
    // int Gsz = gridDim.x;

    for (int i = 0; i < num_idxs; i++){
        int indices_idx = Bidx * Bsz * num_idxs + i * Bsz + tidx;
        indir_idxs[i] = indices[indices_idx];
    }
    for (int i = 0; i < num_idxs; i++){
        int out_idx = Bidx * Bsz * num_idxs + Bsz * i + tidx;
        out[out_idx] = in[indir_idxs[i]];
    }
}

template<typename vt, typename it, int num_idxs>
__global__        
void overlapped_kernel_for_regs(unsigned int idx, vt* in, vt* out, it* indices){
    extern __shared__ int dummy[];
    kernel<vt, it, num_idxs>(idx, in, out, indices);
}

template<typename vt, typename it, int num_idxs>
struct OverlappedIdxDataAccessKernel : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        int N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        // Setup inputs/outputs, both data and indices
        // Can be still in vector form, but is easier to identify explicitly with names   
        // Example
        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vector<it> & indices = super::host_indices[0];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];
        it* & d_indices = super::device_indices_ptrs[0];
        

        int data_reads_per_element = 1; // Actual number 
        int index_reads_per_element = 1; // Actual number
        int writes_per_element = 1; // Actual number

        struct gpu_ctx {
            vt * gpu_in; 
            vt * gpu_out;
            it * gpu_indices;

            __device__        
            void operator() (unsigned int idx){
               extern __shared__ int dummy[];
                    kernel<vt, it, num_idxs>(idx, gpu_in, gpu_out, gpu_indices);
            }
        } ctx ;

        OverlappedIdxDataAccessKernel(int n, int bs, GpuDeviceContext* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 1, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "OverlappedIdxDataAccessKernel";
            this->Gsz /= num_idxs;
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~OverlappedIdxDataAccessKernel(){}

        void init_inputs(bool& pass) override {
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {
            for(int i=0; i<N; ++i){
                indices.push_back(i);
            }
        }

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
            ctx.gpu_indices = d_indices;
        }

        bool local_check_result() override {
            for(int i=0; i<N; ++i){
                if(in[i] != out[i]){
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
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, overlapped_kernel_for_regs<vt,it, num_idxs>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
