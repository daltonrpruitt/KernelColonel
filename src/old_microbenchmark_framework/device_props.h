#pragma once

#include <local_cuda_utils.h>

#include <iostream>
#include <iomanip>

#include <cuda.h>

using std::cout;
using std::endl;

/**
 * @brief Wrapper around cudaDeviceProp
 * 
 * Computes theoretical bandwidth in init()
 * 
 */
class GpuDeviceContext {
    int device_id;
  public:
    cudaDeviceProp props_;
    float theoretical_bw_;

    GpuDeviceContext(int dev_id) : device_id(dev_id) {}
    ~GpuDeviceContext() {}
    
    /**
     * @brief Checks if cuda device available, gets properties, and calculates theoretical bandwidth
     * 
     * Throws excption if cannot properly set the device and get properties.
     */
    void init() {
        cudaErrChk(cudaSetDevice(device_id), "finding GPU device" + std::to_string(device_id));
        cudaErrChk(cudaGetDeviceProperties(&props_, device_id), "getting device properties");
            cout << "Successfully found GPU device "<< props_.name << "\n";
            theoretical_bw_ = (float)props_.memoryClockRate  // kHz = kcycles/s
                            * props_.memoryBusWidth        // bit / cycle
                            / 8                                // byte / 8 bits
                            * 2                                // Assuming Double Data Rate (DDR) memory
                            * 1000 / (1024*1024*1024);                     // 1000 cycles/kcycle * 1 GB/1024^3 B 
			std::streamsize ss = cout.precision();
            cout << "Device '" << props_.name << "' : Max Bandwidth = " << std::fixed << std::setprecision(1) << theoretical_bw_ << " GB/s" << endl;
			cout << std::setprecision(ss) << std::resetiosflags( std::ios::fixed | std::ios::showpoint );
        
        cudaPrintLastError();
    }


};

