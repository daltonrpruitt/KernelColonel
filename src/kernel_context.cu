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

#define DEBUG

using std::string;
using std::cout;
using std::endl;
using std::vector;


template<typename vt, typename it, typename kernel_ctx_t>
__global__
void compute_kernel(int N, kernel_ctx_t ctx) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    ctx(idx);
}



template<typename vt, typename it, int n, int bsz>
struct TestKernelContext {
    public:
        string name = "Array_Copy";
        int N = n;
        int Bsz = bsz;
        int Gsz;


        vector<vector<vt>> host_data{2};
        vector<vt *> device_data_ptrs{2};
        bool okay = true;
        
        // host_vector<it> host_indices;
        // vector<it *> device_indices_ptrs;
        // template<typename vt, typename it>
        struct kernel_context {
            vt * in;
            vt * out;   

            // template<typename vt, typename it>
            __device__        
            void operator() (uint idx){
                // vt* in = data[0];
                // vt* out = data[1];
                out[idx] = in[idx];
            }
        } ctx ;

        void free(){
            for(vt* ptr : device_data_ptrs) cudaFree(ptr);
        }

        TestKernelContext() {
            Gsz = (N+Bsz-1)/Bsz;
            // host_data.resize(2);

            for(int i=0; i<N; ++i){
                host_data[0].push_back(i);
                host_data[1].push_back(0);
            }

            device_data_ptrs.resize(2);
            // vt tmp = 5;
            // device_data_ptrs[0] = &tmp;
            // device_data_ptrs[1] = &tmp;

            bool pass = true;
            
            cudaErrChk(cudaMalloc((void **)&device_data_ptrs[0], N * sizeof(vt)),"device_data_ptrs[0] mem allocation", pass);
            cudaErrChk(cudaMalloc((void **)&device_data_ptrs[1], N * sizeof(vt)),"device_data_ptrs[1] mem allocation", pass);
            

            cudaErrChk(cudaMemcpy(device_data_ptrs[0], host_data[0].data(), N * sizeof(vt), cudaMemcpyHostToDevice), "copy host_data[0] to device_data_ptrs[0]", pass);
            if(!pass) {free(); okay = false;}
            else {
                ctx.in = device_data_ptrs[0];
                ctx.out = device_data_ptrs[1];
            }
        }

        ~TestKernelContext(){
            free();            
        }


        
        void execute() {
            if(!okay) return;
            compute_kernel<vt, it, kernel_context><<<Gsz, Bsz>>>(N, ctx);
            
            bool pass = true;
            cudaErrChk(cudaMemcpy(host_data[1].data(), device_data_ptrs[1], N * sizeof(vt), cudaMemcpyDeviceToHost),"copying device_data_ptrs[1] to host_data[1]", pass);
            if(!pass) {free(); okay = false;}
        }

        bool check_result() {
            if(!okay){
                cout << "Cannot check "<< name << " due to previous failure!" << endl;
                return false;
            };

            for(int i=0; i<N; ++i){
                if(host_data[0][i] != host_data[1][i]){
                    cout << "Validation Failed at " << i << ": in="<<host_data[0][i] << " out="<< host_data[1][i] << endl;
                    return false;
                }
            }
            return true;
        }


};




