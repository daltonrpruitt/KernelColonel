#pragma once 
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
using std::to_string;
using std::cout;
using std::endl;
using std::vector;


template<typename kernel_ctx_t>
__global__
void compute_kernel(int N, kernel_ctx_t ctx) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;
    ctx(idx);
}



template<typename vt, typename it>
struct KernelCPUContext {
    public:
        string name;
        int N=-1;
        int Bsz=-1;
        int Gsz=-1;
        int num_in_data=-1;
        int num_out_data=-1;
        int num_total_data=-1;
        int num_indices=-1;

        vector<vector<vt>> host_data{(unsigned long)num_total_data};
        vector<vt *> device_data_ptrs{(unsigned long)num_total_data};

        bool okay = true;
        
        vector<vector<it>> host_indices{(unsigned long)num_indices};
        vector<it *> device_indices_ptrs{(unsigned long)num_indices};


        void free(){
            for(vt* ptr : device_data_ptrs) cudaFree(ptr);
            for(it* ptr : device_indices_ptrs) cudaFree(ptr);
        }

        virtual void init_inputs() {};
        virtual void init_indices() {};

        KernelCPUContext(int in, int out, int indices, int n, int bs)
            : num_in_data(in), num_out_data(out), num_indices(indices), 
            num_total_data(in+out), N(n), Bsz(bs), Gsz( (n+bs-1)/bs )  {}

        void init(){
            
            init_inputs();
            init_indices();

            device_data_ptrs.resize(num_total_data);

            bool pass = true;
            for(int i=0; i < num_total_data; ++i) {
                cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], N * sizeof(vt)),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                if(!pass) break;
            }
            
            if(pass) {
               for(int i=0; i < num_in_data; ++i) {
                    cudaErrChk(cudaMemcpy(device_data_ptrs[i], host_data[i].data(), N * sizeof(vt), cudaMemcpyHostToDevice), "copy host_data["+to_string(i)+"] to device_data_ptrs["+to_string(i)+"]", pass);                
                    if(!pass) break;
                }
            }

            if(!pass) {free(); okay = false;}
            else { set_dev_ptrs(); }
        }

        ~KernelCPUContext(){
            free();            
        }

        virtual void set_dev_ptrs() {}

        virtual void local_execute() {}

        void execute() {
            if(!okay) return;
            local_execute();
            
            bool pass = true;
            for(int i=num_in_data; i < num_total_data; ++i) {
                cudaErrChk(cudaMemcpy(host_data[i].data(), device_data_ptrs[i], N * sizeof(vt), cudaMemcpyDeviceToHost),"copying device_data_ptrs["+to_string(i)+"] to host_data["+to_string(i)+"]", pass);
            }
            
            if(!pass) {free(); okay = false;}
        }

        virtual bool local_check_result() = 0;

        bool check_result() {
            if(!okay){
                cout << "Cannot check "<< name << " due to previous failure!" << endl;
                return false;
            };
            return local_check_result();
        }

        void run() {
            init();
            execute();
        }
        
        bool run_and_check() {
            run();
            return check_result();     
        }

};
