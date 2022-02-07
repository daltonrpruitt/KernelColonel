#pragma once
#include <cuda.h>
// #include <cuda_runtime_api.h>
#include <string>
using std::string;

void cudaErrChk(cudaError_t status, string msg, bool& pass) {
    if (status != cudaSuccess) {
        std::cerr << "Error with " << msg << "!" << std::endl;
        pass = false;
    }
}

void cudaPrintLastError() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        std::cerr << "Error = " << cudaGetErrorString(status) << std::endl;
}
