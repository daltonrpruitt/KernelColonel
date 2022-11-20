#pragma once 
/**
 * @file kernel_data.cu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Provides context information for GPU kernel execution of driver
 * @version 0.1
 * @date 2022-01-27
 * 
 */

#include <local_cuda_utils.h>
#include <device_props.h>

#include <vector>
#include <algorithm>

#include <cuda.h>

using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;



/**
 * @brief Container for CPU/GPU data for different situations; used for reducing `cudaMemCpy()` calls
 * 
 * Necessary because different kernel contexts may use the same input data, but will the data
 * will be removed and re-computed and copied between separate kernel context instance executions.
 * 
 * @tparam value_t Value type (data arrays)
 * @tparam it Index type (indirection arrays)
 */
template<typename value_t, 
         typename index_t, 
         unsigned int num_in_data, 
         unsigned num_out_data, 
         unsigned int num_indices>
class IKernelData { 
  public:
    IKernelData(unsigned long long n) : N(n) {}

    ~IKernelData(){
        uninit();            
    }

    /**
     * @brief Setup all CPU and GPU data/index arrays
     * 
     * Setup on CPU side, allocate GPU memory, copy data over to GPU. 
     * 
     * @return true Prematurely return if already initialized
     * @return false Failed to initialize properly (handling taken care of by owner of object)
     */
    bool init(int dev_ctx_id){
        if(initialized) { return true; }
        gpu_device_id = dev_ctx_id;

        bool pass = true;
        if(input_size == 0) { input_size = N; }
        if(output_size == 0) { output_size = N; }
        if(indices_size == 0) { indices_size = N; }

        if(pass) init_inputs_cpu();
        if(pass) init_indices_cpu();

        // init inputs/indices gpu...
        if(pass){
            device_data_ptrs.resize(num_total_data);
            cudaErrChk(cudaSetDevice(gpu_device_id), "setting device " + std::to_string(gpu_device_id));
            for(int i=0; i < num_in_data; ++i) {
                cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], input_size * sizeof(value_t)),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                if(!pass) break;
            }

            if(pass) {
                for(int i=num_in_data; i < num_total_data; ++i) {
                    cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], output_size * sizeof(value_t)),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                    if(!pass) break;
                }
            }
            
            if(pass) {
                for(int i=0; i < num_in_data; ++i) {
                    cudaErrChk(cudaMemcpy(device_data_ptrs[i], host_data[i].data(), input_size * sizeof(value_t), cudaMemcpyHostToDevice), "copy host_data["+to_string(i)+"] to device_data_ptrs["+to_string(i)+"]", pass);                
                    if(!pass) break;
                }
            }

            for(int i=0; i < num_indices; ++i) {
                cudaErrChk(cudaMalloc((void **)&device_indices_ptrs[i], indices_size * index_t_size),"device_indices_ptrs["+to_string(i)+"] mem allocation", pass);
                if(!pass) break;
            }

            if(pass) {
                for(int i=0; i < num_indices; ++i) {
                    cudaErrChk(cudaMemcpy(device_indices_ptrs[i], host_indices[i].data(), indices_size * index_t_size, cudaMemcpyHostToDevice), "copy host_indices["+to_string(i)+"] to device_indices_ptrs["+to_string(i)+"]", pass);                
                    if(!pass) break;
                }
            }

            if(pass) { set_dev_ptrs(); }
        }

        if(!pass) {
            free(); 
            okay = false;
            cerr<<"Error in initializing "<<this->name << "for N="<<this->N<<" Bsz="<<this->Bsz;
            if(input_size != 0) cout << " input_sz="<<input_size;
            if(output_size != 0) cout << " output_sz="<<output_size;
            if(indices_size != 0) cout << " indices_sz="<<indices_size;
            cerr << " !" << endl;
        }
        if(pass) initialized = true;
        return pass;
    }

    /**
     * @brief Free relevant structures (CPU and GPU)
     */
    void uninit() {
        if(!initialized) {
            std::cout << "Attempted to uninit() a KernelData instance without being init()'d!" << std::endl;
            return;
        }
        freeGpuData();
        freeCpuData();
        initialized = false;
    }

  private: 
    /**
     * @brief Placeholder for user-defined input data array(s) initialization
     * 
     * Throws an exception if fails.
     */
    virtual void init_inputs_cpu() = 0;

    /**
     * @brief Placeholder for user-defined indices array(s) initialization
     * 
     * Throws an exception if fails.
     */
    virtual void init_indices_cpu() = 0;

    /**
     * @brief Free GPU memory
     */
    void freeGpuData(){
        cudaSetDevice(gpu_device_id);
        for(value_t* ptr : device_data_ptrs)     { cudaFree(ptr); ptr = nullptr; }
        for(index_t* ptr : device_indices_ptrs)  { cudaFree(ptr); ptr = nullptr; }
    }
          
    /**
     * @brief Free CPU memory
     */
    void freeCpuData(){
        for(int i=0; i<num_total_data; ++i) { 
            vector<value_t>().swap(host_data[i]); 
        }
        for(int i=0; i<num_indices; ++i) { 
            vector<index_t>().swap(host_indices[i]); 
        }
    }

    static constexpr unsigned int value_t_size = sizeof(value_t);
    static constexpr unsigned int index_t_size = sizeof(index_t);
    static constexpr unsigned int num_total_data = num_in_data + num_out_data;
    string name;
    
    bool okay = true;
    bool initialized = false;
    
    int gpu_device_id;
protected:
    unsigned long long N=0;
    unsigned long long input_size=0;
    unsigned long long output_size=0;
    unsigned long long indices_size=0;
    vector<vector<value_t>> host_data{(unsigned long)num_total_data};
    vector<value_t *> device_data_ptrs{(unsigned long)num_total_data};
    
    vector<vector<index_t>> host_indices{(unsigned long)num_indices};
    vector<index_t *> device_indices_ptrs{(unsigned long)num_indices};

};
