#pragma once
#include <cuda.h>
// #include <cuda_runtime_api.h>
#include <string>
#include <iostream>

using std::string;
using std::cerr;
using std::endl;

void cudaErrChk(cudaError_t status, string msg, bool& pass) {
    if (status != cudaSuccess) {
        cerr << "Error with " << msg << "!" << endl;
        pass = false;
    }
}

void cudaErrChk(cudaError_t status, string msg) {
    if (status != cudaSuccess) {
        std::string cuda_error_string = cudaGetErrorString(status);
        throw std::runtime_error("Error with " + msg +  " : cuda err msg= '" + cuda_error_string + "'");
    }
}

void cudaPrintLastError() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        cerr << "Error = " << cudaGetErrorString(status) << endl;
}
