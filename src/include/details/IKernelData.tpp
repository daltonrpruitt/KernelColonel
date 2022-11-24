#pragma once
/**
 * @file IKernelData.tcu
 * @author Dalton Winans-Pruitt (daltonrpruitt@gmail.com)
 * @brief Implements IKernelData
 * 
 */

#include "IKernelData.hpp"

#include <utils/local_cuda_utils.h>

#include <algorithm>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace KernelColonel {

using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::vector;




template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::IKernelData(unsigned long long n) : N(n) {}

template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::~IKernelData(){
    uninit();            
    }

template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
bool IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::init(int dev_ctx_id){
    if(state == DataState::INIT){
        if(dev_ctx_id == gpu_device_id) {
            cout <<"Tried to reinitialize "<< this->name << "with same device id=" << gpu_device_id << "; Ignored" << std::endl;
            return true;
        } else {
            cerr<<"Tried to reinitialize "<< this->name << "with current device id=" << gpu_device_id <<" and new device ID="<< dev_ctx_id << "; Not allowed!" << std::endl;
            return false;
        }
    }
    gpu_device_id = dev_ctx_id;

    bool pass = true;
    if(input_size == 0) { input_size = N; }
    if(output_size == 0) { output_size = N; }
    if(indices_size == 0) { indices_size = N; }

    if(pass) initInputsCpu();
    if(pass) initIndicesCpu();

    // init inputs/indices gpu...
    if(pass){
        cudaErrChk(cudaSetDevice(gpu_device_id), "setting device " + std::to_string(gpu_device_id));
        for(int i=0; i < num_in_data; ++i) {
            cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], input_size * value_t_size),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
            if(!pass) break;
        }

        if(pass) {
            for(int i=num_in_data; i < num_total_data; ++i) {
                cudaErrChk(cudaMalloc((void **)&device_data_ptrs[i], output_size * value_t_size),"device_data_ptrs["+to_string(i)+"] mem allocation", pass);
                if(!pass) break;
            }
        }
        
        if(pass) {
            for(int i=0; i < num_in_data; ++i) {
                cudaErrChk(cudaMemcpy(device_data_ptrs[i], host_data[i].data(), input_size * value_t_size, cudaMemcpyHostToDevice), "copy host_data["+to_string(i)+"] to device_data_ptrs["+to_string(i)+"]", pass);                
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

        if(pass) { setGpuNamedData(); }
    }

    if(!pass) {
        freeGpuData();
        freeCpuData();
        okay = false;
        cerr<<"Error in initializing "<<this->name << "for N="<<this->N;
        if(input_size != 0) cout << " input_sz="<<input_size;
        if(output_size != 0) cout << " output_sz="<<output_size;
        if(indices_size != 0) cout << " indices_sz="<<indices_size;
        cerr << " !" << endl;
    }
    if(pass) state = DataState::INIT;
    return pass;
}

template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
void IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::uninit() {
    if(state != DataState::INIT) {
        std::cout << "Attempted to uninit() a KernelData instance without being init()'d!" << std::endl;
        return;
    }
    freeGpuData();
    freeCpuData();
    gpu_named_data = gpu_data_s_t();
    state = DataState::UNINIT;
}
    
template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
void IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::copyOutputToDevice(){
    for(int i=num_in_data; i < num_total_data; ++i) {
        cudaErrChk(cudaMemcpy(host_data[i].data(), device_data_ptrs[i], output_size * value_t_size, cudaMemcpyDeviceToHost),"copying device_data_ptrs["+to_string(i)+"] to host_data["+to_string(i)+"]");
    }            
}
    
template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
void IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::freeGpuData(){
    cudaSetDevice(gpu_device_id);
    for(auto &ptr : device_data_ptrs)     { cudaFree(ptr); ptr = nullptr; }
    for(auto &ptr : device_indices_ptrs)  { cudaFree(ptr); ptr = nullptr; }
}
        
template<typename vt, typename it, unsigned int num_in_data, unsigned int num_out_data, unsigned int num_indices, typename gpu_data_s_t>
void IKernelData<vt,it,num_in_data,num_out_data,num_indices,gpu_data_s_t>::freeCpuData(){
    for(int i=0; i<num_total_data; ++i) { 
        vector<vt>().swap(host_data[i]); 
    }
    for(int i=0; i<num_indices; ++i) { 
        vector<it>().swap(host_indices[i]); 
    }
}

} // namespace KernelColonel