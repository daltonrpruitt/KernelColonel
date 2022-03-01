#pragma once
/**
 * @file interleaved_copy.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from ArrayCopyContext; testing for burst mode 
 * @version 0.1
 * @date 2022-02-24
 * 
 * Based on my write up that is currently at 
 * https://app.diagrams.net/#G1NgMgo7joWueBNKOvifh_rLW__W7VOhrP
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

template<typename vt, typename it, int block_life, int elements>
__forceinline__ __host__ __device__        
void interleaved_kernel(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){

    unsigned long long b_idx = blockIdx.x;
    unsigned long long t_idx = threadIdx.x;
    unsigned long long Bsz = blockDim.x;
    unsigned long long Gsz = gridDim.x;
    
    for(int x=0; x < block_life; ++x) {
        for(int y=0; y < elements; ++y) {
            unsigned long long data_idx =  b_idx * Bsz * elements +
                    t_idx + Gsz * Bsz * elements * x + Bsz*y;
            if(data_idx >= N) continue;
            gpu_out[data_idx] = gpu_in[data_idx];
        }
    }
}


template<typename vt, typename it, int block_life, int elements>
__global__        
void uncoalesced_reuse_kernel_for_regs(uint idx, vt* gpu_in, vt* gpu_out, unsigned long long N){
        extern __shared__ int dummy[];
        interleaved_kernel<vt, it, block_life, elements>(idx, gpu_in, gpu_out, N);
}

template<typename vt, typename it, int block_life, int elements>
struct InterleavedCopyContext : public KernelCPUContext<vt, it> {
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
            void operator() (uint idx){
                extern __shared__ int dummy[];
                interleaved_kernel<vt, it, block_life, elements>(idx, gpu_in, gpu_out, N);
            }
        } ctx ;

        InterleavedCopyContext(int n, int bs, device_context* dev_ctx, int shd_mem_alloc=0) 
            : super(1, 1, 0, n, bs, dev_ctx, shd_mem_alloc) {
            assert(N % (block_life * elements) == 0);
            this->name = "InterleavedCopy"; 
            this->Gsz /=  block_life * elements;
            assert(this->Gsz > 0);
            this->total_data_reads = N * data_reads_per_element;
            this->total_index_reads = N * index_reads_per_element;
            this->total_writes = N * writes_per_element;
        }
        ~InterleavedCopyContext(){}

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
            cout << "InterleavedCopy with : "
                 <<" Block life=" << block_life 
                 << " Local group size=" << local_group_size 
                 << " Elements per group=" << elements << endl;
        }

        float local_execute() override {
            if(this->dev_ctx->props_.major >= 7) {
                //cout << "this->dev_ctx->props_.sharedMemPerMultiprocessor = " << this->dev_ctx->props_.sharedMemPerMultiprocessor << endl;
                //cout << "this->shared_memory_usage = " << this->shared_memory_usage << endl;
                //cout << "cudaDevAttrMaxSharedMemoryPerBlockOptin = " << this->dev_ctx->props_.sharedMemPerBlockOptin << endl;
                cudaFuncAttributes attr;
                cudaFuncGetAttributes(&attr, compute_kernel<gpu_ctx>);
                //cout << " sharedSizeBytes = " << attr.sharedSizeBytes << endl;
                int shmem = this->dev_ctx->props_.sharedMemPerMultiprocessor-1024-attr.sharedSizeBytes;
                cudaFuncSetAttribute(compute_kernel<gpu_ctx>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
                cudaPrintLastError();
            }

            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);

            cudaEventRecord(start);
            compute_kernel<gpu_ctx><<<Gsz, Bsz, this->shared_memory_usage>>>(N, ctx);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaPrintLastError();

            float time = 0;
            cudaEventElapsedTime(&time, start, stop);
            cudaEventDestroy(start); cudaEventDestroy(stop);
            return time; 
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
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *uncoalesced_reuse_kernel_for_regs<vt,it,block_life,elements>), "getting function attributes (for # registers)", pass);
            if(!pass) {
                this->okay = false; 
                return;
            }
            this->register_usage = funcAttrib.numRegs;
        }

};
