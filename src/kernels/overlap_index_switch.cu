#pragma once
/**
 * @file overlap_index_switch.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Load next index before current data
 * @version 0.1
 * @date 2022-02-14
 * 
 * Test latency amortization technique of collapsing just one latency 
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


template<typename vt, typename it, int num_idxs, int num_elements>
__forceinline__ __host__ __device__        
void kernel(uint idx, vt* in, vt* out, it* indices){
    static_assert(num_elements % num_idxs == 0);

    it indir_idxs_1[num_idxs];
    it indir_idxs_2[num_idxs];
    
    it * curr_idxs = indir_idxs_2;
    it * next_idxs = indir_idxs_1;
    it * tmp;


    uint tidx = threadIdx.x;
    uint Bidx = blockIdx.x;

    uint Bsz = blockDim.x;
    // int Gsz = gridDim.x;
    for(int j=0; j < num_idxs; ++j) { 
        next_idxs[j] = Bidx * Bsz * num_elements +  j * Bsz + tidx;
    } 

    for(int i=0; i < num_elements / num_idxs; ++i) {
        tmp = curr_idxs; 
        curr_idxs = next_idxs; 
        next_idxs = tmp; 

        uint curr_idx_base = Bidx * Bsz * ( num_idxs + i ) + tidx;

#pragma unroll (num_idxs)
        for (int j = 0; j < num_idxs; j++){
            next_idxs[j] = indices[curr_idx_base + (j+1) * Bsz];
        }
#pragma unroll (num_idxs)
        for (int j = 0; j < num_idxs; j++){
            // int out_idx = Bidx * Bsz * num_idxs + Bsz * j + tidx;
            out[curr_idx_base + j * Bsz] = in[curr_idxs[j]];
        }
    }
}

template<typename vt, typename it, int num_idxs, int num_elements>
__global__        
void overlapped_kernel_for_regs(uint idx, vt* in, vt* out, it* indices){
    extern __shared__ int dummy[];
    kernel<vt, it, num_idxs, num_elements>(idx, in, out, indices);
}

template<typename vt, typename it, int num_idxs, int num_elements>
struct OverlappedIndexSwitchKernel : public KernelCPUContext<vt, it> {
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
        int index_reads_per_element = 1; // Actual number
        int writes_per_element = 1; // Actual number

        struct gpu_ctx {
            vt * gpu_in; 
            vt * gpu_out;
            it * gpu_indices;

            __device__        
            void operator() (uint idx){
               extern __shared__ int dummy[];
                    kernel<vt, it, num_idxs, num_elements>(idx, gpu_in, gpu_out, gpu_indices);
            }
        } ctx ;

        OverlappedIndexSwitchKernel(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 1, n, bs, dev_ctx, shd_mem_alloc) {
            this->name = "OverlappedIndexSwitchKernel";
            this->Gsz /= num_idxs*num_elements;
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~OverlappedIndexSwitchKernel(){}

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
            bool pass = true;
            for(int i=0; i<N; ++i){
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
                for(int i=0; i < output_size; ++i) { 
                    cout << std::setw(10) << in[i] <<"|" <<std::setw(10)<<out[i] << endl; 
                }
                cout << std::setw(10) << "..." << endl;
                for(int i=this->Bsz-output_size/2; i < this->Bsz+output_size/2; ++i) { 
                    cout << std::setw(10) << in[i] <<"|" <<std::setw(10)<<out[i] << endl; 
                }
            }
            return pass;
        }

        // No change
        void local_execute() override {
            if(this->dev_ctx->props_.major >= 7) {
                cudaFuncSetAttribute(compute_kernel<gpu_ctx>, cudaFuncAttributeMaxDynamicSharedMemorySize, this->dev_ctx->props_.sharedMemPerMultiprocessor);
            }
            compute_kernel<gpu_ctx><<<Gsz, Bsz, this->shared_memory_usage>>>(N, ctx);
        }

        // No change
        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *overlapped_kernel_for_regs<vt,it, num_idxs, num_elements>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
