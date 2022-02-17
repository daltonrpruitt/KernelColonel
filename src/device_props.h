#pragma once

#include <local_cuda_utils.h>

#include <iostream>

using std::pow;
using std::cout; 
using std::endl;

class device_context {
  public:
    cudaDeviceProp props_;
    float theoretical_bw_;

    device_context() {}
    ~device_context() {}
    
    bool init() {
        bool pass = true;
        cudaErrChk(cudaSetDevice(0), "finding GPU device", pass);
        if(pass) { cudaErrChk(cudaGetDeviceProperties(&props_, 0), "getting device properties", pass); }
        if(pass) { 
            cout << "Successfully found GPU device "<< props_.name << endl;
            theoretical_bw_ = (float)props_.memoryClockRate  // kHz = kcycles/s
                            * props_.memoryBusWidth        // bit / cycle
                            / 8                                // byte / 8 bits
                            * 2                                // Assuming Double Data Rate (DDR) memory
                            * pow(1000.0, -2);                     // 1000 cycles/kcycle * 1 GB/10^9 B = 1/10^6
        }
        cudaPrintLastError();
        return pass;
    }


};

