#pragma once
/**
 * @file copy.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from KernelCPUContext; actual kernel specification
 * @version 0.1
 * @date 2022-01-27
 * 
 */

#include <vector>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it>
__global__        
void kernel (uint idx, vt* gpu_in, vt* gpu_out){
        gpu_out[idx] = gpu_in[idx];
}

template<typename vt, typename it>
struct ArrayCopyContext : public KernelCPUContext<vt, it> {
    public:
        typedef KernelCPUContext<vt, it> super;
        // name = "Array_Copy";
        int N = super::N;
        int Gsz = super::Gsz;
        int Bsz = super::Bsz;

        vector<vt> & in = super::host_data[0];
        vector<vt> & out = super::host_data[1];
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];

        int reads_per_element = 1;
        int writes_per_element = 1;
        int total_reads;
        int total_writes;
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;   

            __device__        
            void operator() (uint idx){
                gpu_out[idx] = gpu_in[idx];
            }
        } ctx ;

        ArrayCopyContext(int n, int bs, device_context dev_ctx) 
            : super(1, 1, 0, n, bs, dev_ctx) {
            this->name = "Array_Copy_N=" +std::to_string(n) + "_Bs="+std::to_string(bs);
            total_reads = N * reads_per_element;
            total_writes = N * writes_per_element;
        }
        ~ArrayCopyContext(){}

        void init_inputs() override {
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices() override {}

        void set_dev_ptrs() override {
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
        }

        
        void local_execute() override {
            compute_kernel<gpu_ctx><<<Gsz, Bsz>>>(N, ctx);
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

        void local_compute_register_usage() override {
            bool pass = true;
            
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel<vt,it>), "Getting function attributes (for # registers)", pass);
            this->register_usage = funcAttrib.numRegs;
#ifdef DEBUG
            cout << this->name << " numRegs=" << this->register_usage << endl;
#endif
        }

};
