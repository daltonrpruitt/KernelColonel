/**
 * @file kernel_context.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides context information for GPU kernel execution of driver
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

        

        
        struct gpu_ctx {
            vt * gpu_in;
            vt * gpu_out;   

            __device__        
            void operator() (uint idx){
                gpu_out[idx] = gpu_in[idx];
            }
        } ctx ;

        ArrayCopyContext(int n, int bs) 
            : super(1, 1, 0, n, bs) {
            this->name = "Array_Copy";
        }
        ~ArrayCopyContext(){}

        void init_inputs() override {
            for(int i=0; i<N; ++i){
                this->in.push_back(i);
                this->out.push_back(0);
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
};
