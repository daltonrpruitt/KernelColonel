#pragma once

#include <local_cuda_utils.h>

#include <iostream>
#include <iomanip>

#include <cuda.h>

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
                            * 1000 / (1024*1024*1024);                     // 1000 cycles/kcycle * 1 GB/1024^3 B 
			std::streamsize ss = cout.precision();
            cout << "Device '" << props_.name << "' : Max Bandwidth = " << std::fixed << std::setprecision(1) << theoretical_bw_ << " GB/s" << endl;
			cout << std::setprecision(ss) << resetiosflags( std::ios::fixed | std::ios::showpoint );
        }
        cudaPrintLastError();
        return pass;
    }


};

