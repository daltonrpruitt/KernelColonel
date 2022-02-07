#pragma once
#include <cuda.h>
// #include <cuda_runtime_api.h>
#include <string>
#include <iostream>

using namespace std;

void cudaErrChk(cudaError_t status, string msg, bool& pass) {
    if (status != cudaSuccess) {
        cerr << "Error with " << msg << "!" << endl;
        pass = false;
    }
}

void cudaPrintLastError() {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        cerr << "Error = " << cudaGetErrorString(status) << endl;
}
