#pragma once
/**
 * @file simple_indirection.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2022-02-08
 * 
 * Based on copy.cu 
 * Most of what needs to be changed (or not) is listed.
 * Do not modify function signatures unless obvious (comment in signature).
 */

#include <vector>
#include <type_traits>

#include <cuda.h>
#include <local_cuda_utils.h>
#include <kernel_context.cu>

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;

template<typename vt, typename it>
__forceinline__ __host__ __device__        
void kernel_direct(uint idx, vt* in, vt* out, it* indices){
    it indirect_idx = indices[idx];
    if(indirect_idx < -1) {return;} // ensure read in indirection 
    out[idx] = in[idx];
}

template<typename vt, typename it>
__forceinline__ __host__ __device__        
void kernel_indirect(uint idx, vt* in, vt* out, it* indices){
    it indirect_idx = indices[idx];
    if(indirect_idx < -1) {return;}
    out[idx] = in[indirect_idx];
}

template<typename vt, typename it, bool is_indirect>
__global__        
void kernel_for_regs(uint idx, vt* in, vt* out, it* indices){
    if constexpr(is_indirect) {
        kernel_indirect<vt, it>(idx, in, out, indices);
    } else {
        kernel_direct<vt, it>(idx, in, out, indices);
    }
}

template<typename vt, typename it, bool is_indirect>
struct SimpleIndirectionKernel : public KernelCPUContext<vt, it> {
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
        

        int reads_per_element = 1; // Actual number 
        int writes_per_element = 1; // Actual number
        int indirect_reads_per_element = 1; // Actual number
        int total_reads;
        int total_writes;
        int total_indirect_reads;

        struct gpu_ctx {
            vt * gpu_in; 
            vt * gpu_out;
            it * gpu_indices;

            __device__        
            void operator() (uint idx){
                if constexpr(is_indirect) {
                    kernel_indirect<vt, it>(idx, gpu_in, gpu_out, gpu_indices);
                } else {
                    kernel_direct<vt, it>(idx, gpu_in, gpu_out, gpu_indices);
                }
            }
        } ctx ;

        SimpleIndirectionKernel(int n, int bs, device_context dev_ctx) 
            : super(1, 1, 1, n, bs, dev_ctx) {
            if(is_indirect){
                this->name = "SimpleIndirectionTest_Indirect";
            } else {
                this->name = "SimpleIndirectionTest_Direct";
            }
            total_reads = N * reads_per_element;
            total_writes = N * writes_per_element;
            total_indirect_reads = N * indirect_reads_per_element;
        }
        ~SimpleIndirectionKernel(){}

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
        void local_execute() override {
            compute_kernel<gpu_ctx><<<Gsz, Bsz>>>(N, ctx);
        }

        // No change
        void local_compute_register_usage(bool& pass) override {   
            // Kernel Registers 
            struct cudaFuncAttributes funcAttrib;
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel_for_regs<vt,it, is_indirect>), "getting function attributes (for # registers)", pass);
            if(!pass) return;
            this->register_usage = funcAttrib.numRegs;
#ifdef DEBUG
            cout << this->name << " numRegs=" << this->register_usage << endl;
#endif
        }

};
