#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>

inline void cudaErrChk(cudaError_t status, std::string msg, bool& pass) {
    if (status != cudaSuccess) {
        std::cerr << "Error with " << msg << "!" << std::endl;
        pass = false;
    }
}

inline void cudaErrChk(cudaError_t status, std::string msg) {
    if (status != cudaSuccess) {
        std::string cuda_error_string = cudaGetErrorString(status);
        throw std::runtime_error("Error with " + msg +  " : cuda err msg= '" + cuda_error_string + "'");
    }
}

inline void cudaPrintLastError() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        std::cerr << "Error = " << cudaGetErrorString(status) << std::endl;
}
