#pragma once
/**
 * @file template.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Derived from KernelCPUContext; what to start with when making new kernel
 * @version 0.1
 * @date 2022-02-08
 * 
 * Based on copy.cu 
 * Most of what needs to be changed (or not) is listed.
 * Do not modify function signatures unless obvious (comment in signature).
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
__forceinline__ __host__ __device__        
void kernel(uint idx, /* parameters */){
    /* 
     * Actual kernel code
     * The idx above is the thread index
    */
}


template<typename vt, typename it>
__global__        
void kernel_for_regs(uint idx, /* parameters (same as kernel) */){
        kernel<vt, it>(idx, /* pass parameters from above */);
}

template<typename vt, typename it>
struct TemplateKernelContext : public KernelCPUContext<vt, it> {
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
        vt* & d_in = super::device_data_ptrs[0];
        vt* & d_out = super::device_data_ptrs[1];
        

        int reads_per_element = -1; // Actual number 
        int writes_per_element = -1; // Actual number
        int total_reads;
        int total_writes;
        struct gpu_ctx {
            // Params for kernel; set in set_dev_ptrs()
            vt * gpu_in; 
            vt * gpu_out;   

            __device__        
            void operator() (uint idx){
                kernel<vt, it>(idx, /* pass parameters (same as before) */);
            }
        } ctx ;

        TemplateKernelContext(int n, int bs, device_context dev_ctx) 
            : super(/*#inputs, #outputs, #index arrays */, n, bs, dev_ctx) {
            this->name = "/* Identifying name */";
            total_reads = N * reads_per_element;
            total_writes = N * writes_per_element;
        }
        ~TemplateKernelContext(){}

        void init_inputs(bool& pass) override {
            // Initialize inputs (data)
            // Example
            for(int i=0; i<N; ++i){
                in.push_back(i);
                out.push_back(0);
            }
        }

        void init_indices(bool& pass) override {
            // Initialize indices (not necessary if not using indirection)
            // Example
            for(int i=0; i<N; ++i){
                in_out_mapping.push_back(i);
            }
        }

        void set_dev_ptrs() override {
            // Set device pointers of gpu_ctx above
            // Should come from member vars listed above
            // Example
            ctx.gpu_in = d_in;
            ctx.gpu_out = d_out;
        }

        bool local_check_result() override {
            // Verify that the output of kernel is correct
            // Base class copies the data back to host, so work with 
            // the host member vars defined above
            // Example for simple copy kernel
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
            cudaErrChk(cudaFuncGetAttributes(&funcAttrib, *kernel_for_regs<vt,it>), "getting function attributes (for # registers)", pass);
            if(!pass) return;
            this->register_usage = funcAttrib.numRegs;
#ifdef DEBUG
            cout << this->name << " numRegs=" << this->register_usage << endl;
#endif
        }

};
